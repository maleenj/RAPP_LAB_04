// ParticleFlowField.cs — Example 01 (special): a living particle cloud sculpted by
// the body. The 18 skeleton points act as attractors in a GPU particle system:
// particles drift, get pulled toward the moving points and pushed away when too
// close, and swirl with turbulence that grows where the body moves fast. You never
// see the skeleton — only the current it creates.
//
// Self-contained GPU compute system (no VFX Graph needed). Built-in Render Pipeline.
//
// SETUP (see the README in this folder for the full version):
//   1. Add this component to an empty GameObject. An EnactData is added with it —
//      on that EnactData set Channel = "skeleton".
//   2. Assign the two assets below in the Inspector:
//        Sim Shader    -> ParticleFlowField.compute
//        Render Shader -> ParticleFlowField (Enact/ParticleFlowField)
//   3. Press Play. Tune the dials. Use a FlyCamera to move around it.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class ParticleFlowField : MonoBehaviour
{
    [Header("Assets (assign these)")]
    [Tooltip("ParticleFlowField.compute")]
    public ComputeShader simShader;
    [Tooltip("The Enact/ParticleFlowField shader")]
    public Shader renderShader;

    [Header("Particles")]
    [Range(1000, 1000000)] public int particleCount = 150000;
    [Tooltip("Seconds a particle lives before respawning near the body.")]
    public float particleLife = 4f;
    [Tooltip("Sprite size in metres.")]
    public float particleSize = 0.03f;

    [Header("Forces — pull / push")]
    public float attractStrength = 6f;
    [Tooltip("Higher = pull only acts up close (tighter cloud).")]
    public float attractFalloff = 1.5f;
    public float repelRadius = 0.12f;
    public float repelStrength = 3f;

    [Header("Forces — turbulence")]
    public float turbulence = 1.5f;
    [Tooltip("Spatial frequency of the swirl (higher = finer eddies).")]
    public float turbulenceScale = 0.7f;
    public float turbulenceSpeed = 0.25f;
    [Tooltip("Extra turbulence added where the body moves fast.")]
    public float velocityTurbulence = 5f;
    [Tooltip("Velocity damping per 1/60s (0.85 = heavy drag, 0.98 = floaty).")]
    [Range(0.5f, 0.995f)] public float drag = 0.9f;

    [Header("Emission")]
    [Tooltip("OFF (recommended): particles fill the whole volume and the body just " +
             "sculpts them into a flowing current (no clump). ON: particles are born " +
             "at the body and flow outward.")]
    public bool emitFromBody = false;
    [Tooltip("Used only when Emit From Body is ON — spawn radius around a body point.")]
    public float spawnRadius = 0.3f;
    [Tooltip("Initial outward speed of new particles, so they flow instead of piling up.")]
    public float spawnSpeed = 0.2f;

    [Header("Look — colour & glow")]
    public Gradient colors;
    [Tooltip("Particle speed that maps to the END of the gradient.")]
    public float speedForMaxColor = 3f;
    [Tooltip("HDR brightness multiplier (needs HDR/Bloom for full effect).")]
    public float intensity = 2f;
    [Range(0.2f, 6f)] public float edgeSoftness = 2f;

    [Header("Coordinate mapping (match SkeletonVisualizer)")]
    public bool rosToUnity = true;
    public float scale = 1f;
    public Vector3 offset = Vector3.zero;
    [Tooltip("Treat exact (0,0,0) keypoints as 'missing' and exclude them (matches " +
             "SkeletonVisualizer). Dropped joints sit at the camera origin, so leaving " +
             "this OFF lets them yank the cloud/centroid toward (0,0,0).")]
    public bool skipZeroPoints = true;
    [Tooltip("How quickly attractors chase the latest skeleton (higher = snappier).")]
    public float pointResponse = 12f;
    [Tooltip("Cloud volume radius (also: particles further than this recycle).")]
    public float boundsRadius = 8f;

    public enum Axis { X, Y, Z }

    [Header("Move the source with the actor")]
    [Tooltip("ON: slide the cloud along a line as the actor moves across the stage. " +
             "OFF: cloud sits at the body in 3D.")]
    public bool trackPerson = false;
    [Tooltip("Physical width of the actor's interaction area, in metres (e.g. 7).")]
    public float interactionWidth = 7f;
    [Tooltip("Which world axis is the actor's left-right movement (X with default rosToUnity).")]
    public Axis stageHorizontalAxis = Axis.X;
    [Tooltip("Stage coordinate on that axis that maps to the CENTRE of the track.")]
    public float horizontalCenter = 0f;
    [Tooltip("World position of the cloud when the actor is centred — set the depth & height here.")]
    public Vector3 trackCenter = Vector3.zero;
    [Tooltip("World direction the cloud slides along (default = screen-horizontal, +X).")]
    public Vector3 trackAxis = Vector3.right;
    [Tooltip("Total distance (world metres) the cloud travels across the full stage width.")]
    public float trackWidth = 7f;
    [Tooltip("ON (recommended): slide the cloud using the stable torso joints (neck + " +
             "hips) instead of the full-body average — far less jitter from swinging " +
             "arms/legs. OFF: use the average of every visible joint.")]
    public bool stableTracking = true;

    const int MAX_ATTRACTORS = 64;
    // Stable mid-body keypoints for tracking (ZED BODY_18): neck, right hip, left hip.
    static readonly int[] TORSO_KEYS = { 1, 8, 11 };
    const int PARTICLE_STRIDE = sizeof(float) * 8;   // float3 pos, float3 vel, life, seed

    EnactData _data;
    ComputeBuffer _particles;
    ComputeBuffer _attractors;
    Vector4[] _attractorData = new Vector4[MAX_ATTRACTORS];
    Vector3[] _smoothed = new Vector3[MAX_ATTRACTORS];
    Vector3[] _prevSmoothed = new Vector3[MAX_ATTRACTORS];
    bool[] _jointValid = new bool[MAX_ATTRACTORS];   // visible (non-zero) this frame
    bool[] _jointSeen = new bool[MAX_ATTRACTORS];    // ever seen -> smoothing initialised
    int _activeCount;
    Vector3 _centroid;
    float _meanSpeed;

    Material _mat;
    Texture2D _ramp;
    int _initKernel, _simKernel;
    int _allocated;

    void Reset()
    {
        colors = DefaultGradient();
    }

    void OnEnable()
    {
        _data = GetComponent<EnactData>();
        if (colors == null || colors.colorKeys.Length == 0) colors = DefaultGradient();
        if (simShader == null || renderShader == null)
        {
            Debug.LogError("[ParticleFlowField] Assign Sim Shader (.compute) and Render Shader in the Inspector.");
            enabled = false;
            return;
        }
        _initKernel = simShader.FindKernel("Init");
        _simKernel = simShader.FindKernel("Simulate");
        _mat = new Material(renderShader);
        BuildRamp();
        Allocate(particleCount);
    }

    void OnDisable() => ReleaseAll();

    void Allocate(int count)
    {
        ReleaseBuffers();
        count = Mathf.Max(64, count);
        _particles = new ComputeBuffer(count, PARTICLE_STRIDE);
        _attractors = new ComputeBuffer(MAX_ATTRACTORS, sizeof(float) * 4);
        _allocated = count;

        // seed the particles
        simShader.SetInt("_ParticleCount", count);
        simShader.SetFloat("_ParticleLife", particleLife);
        simShader.SetFloat("_SpawnRadius", spawnRadius);
        simShader.SetFloat("_SpawnSpeed", spawnSpeed);
        simShader.SetFloat("_EmitFromBody", emitFromBody ? 1f : 0f);
        simShader.SetInt("_AttractorCount", 0);
        simShader.SetVector("_BoundsCenter", Vector3.zero);
        simShader.SetFloat("_BoundsRadius", boundsRadius);
        simShader.SetFloat("_TimeY", 0f);
        simShader.SetBuffer(_initKernel, "_Particles", _particles);
        simShader.SetBuffer(_initKernel, "_Attractors", _attractors);
        simShader.Dispatch(_initKernel, Groups(count), 1, 1);

        _mat.SetBuffer("_Particles", _particles);
        System.Array.Clear(_jointSeen, 0, _jointSeen.Length);
        System.Array.Clear(_jointValid, 0, _jointValid.Length);
    }

    void Update()
    {
        if (_particles == null) return;
        if (particleCount != _allocated) Allocate(particleCount);

        float dt = Mathf.Min(Time.deltaTime, 0.05f);

        UpdateAttractors(dt);

        // --- simulate on GPU ---
        simShader.SetInt("_ParticleCount", _allocated);
        simShader.SetInt("_AttractorCount", _activeCount);
        simShader.SetFloat("_DeltaTime", dt);
        simShader.SetFloat("_TimeY", Time.time);
        simShader.SetFloat("_AttractStrength", attractStrength);
        simShader.SetFloat("_AttractFalloff", attractFalloff);
        simShader.SetFloat("_RepelRadius", repelRadius);
        simShader.SetFloat("_RepelStrength", repelStrength);
        simShader.SetFloat("_Turbulence", turbulence);
        simShader.SetFloat("_TurbScale", turbulenceScale);
        simShader.SetFloat("_TurbSpeed", turbulenceSpeed);
        simShader.SetFloat("_VelTurbulence", velocityTurbulence);
        simShader.SetFloat("_MeanSpeed", _meanSpeed);
        simShader.SetFloat("_Drag", drag);
        simShader.SetFloat("_SpawnRadius", spawnRadius);
        simShader.SetFloat("_SpawnSpeed", spawnSpeed);
        simShader.SetFloat("_EmitFromBody", emitFromBody ? 1f : 0f);
        simShader.SetFloat("_ParticleLife", particleLife);
        simShader.SetVector("_BoundsCenter", _centroid);
        simShader.SetFloat("_BoundsRadius", boundsRadius);
        simShader.SetBuffer(_simKernel, "_Particles", _particles);
        simShader.SetBuffer(_simKernel, "_Attractors", _attractors);
        simShader.Dispatch(_simKernel, Groups(_allocated), 1, 1);

        // --- draw ---
        _mat.SetTexture("_ColorRamp", _ramp);
        _mat.SetFloat("_Size", particleSize);
        _mat.SetFloat("_Intensity", intensity);
        _mat.SetFloat("_SpeedMax", speedForMaxColor);
        _mat.SetFloat("_LifeMax", particleLife);
        _mat.SetFloat("_Softness", edgeSoftness);

        var bounds = new Bounds(_centroid, Vector3.one * boundsRadius * 2.5f);
        Graphics.DrawProcedural(_mat, bounds, MeshTopology.Triangles, _allocated * 6);
    }

    void UpdateAttractors(float dt)
    {
        if (_data == null || !_data.HasData || _data.Frame == null || _data.Frame.Rows <= 0)
        {
            // no skeleton yet: keep last attractors but decay the turbulence boost
            _meanSpeed = Mathf.Lerp(_meanSpeed, 0f, 0.1f);
            return;
        }

        int rows = Mathf.Min(_data.Frame.Rows, MAX_ATTRACTORS);
        var f = _data.Frame;
        float follow = 1f - Mathf.Exp(-pointResponse * dt);
        Vector3 sum = Vector3.zero;     // centroid of VISIBLE body points (Unity space)
        float speedSum = 0f;
        int n = 0;                      // number of valid attractors uploaded this frame

        for (int i = 0; i < rows; i++)
        {
            float rx = f.Get(i, 0), ry = f.Get(i, 1), rz = f.Get(i, 2);
            bool missing = skipZeroPoints && rx == 0f && ry == 0f && rz == 0f;
            _jointValid[i] = !missing;
            if (missing) continue;      // dropped joint sits at the origin — ignore it

            Vector3 target = ToUnity(rx, ry, rz);
            // Snap on first sighting / reappearance so a stale smoothed value doesn't
            // drag a freshly-detected joint across the scene.
            if (!_jointSeen[i]) { _smoothed[i] = target; _prevSmoothed[i] = target; _jointSeen[i] = true; }
            else                 _smoothed[i] = Vector3.Lerp(_smoothed[i], target, follow);

            float spd = (_smoothed[i] - _prevSmoothed[i]).magnitude / Mathf.Max(dt, 1e-4f);
            _attractorData[n] = new Vector4(_smoothed[i].x, _smoothed[i].y, _smoothed[i].z, spd);
            _prevSmoothed[i] = _smoothed[i];
            sum += _smoothed[i];
            speedSum += spd;
            n++;
        }

        if (n == 0)   // every joint dropped this frame: hold position, decay the boost
        {
            _meanSpeed = Mathf.Lerp(_meanSpeed, 0f, 0.1f);
            return;
        }

        _activeCount = n;
        _meanSpeed = speedSum / n;

        // Where should the whole cloud sit? Either at the body (3D), or slid along a
        // line in the scene based on where the actor is across the stage.
        Vector3 actorCentroid = sum / n;
        Vector3 cloudCenter = trackPerson ? SourcePosition(actorCentroid) : actorCentroid;
        Vector3 off = cloudCenter - actorCentroid;
        if (off.sqrMagnitude > 1e-10f)
        {
            Vector4 o4 = new Vector4(off.x, off.y, off.z, 0f);
            for (int i = 0; i < n; i++) _attractorData[i] += o4;   // translate body -> backdrop
        }

        _centroid = cloudCenter;
        _attractors.SetData(_attractorData);
    }

    // Slide the cloud along a line: the actor's position across the stage
    // (interactionWidth) maps to a position along trackAxis, of length trackWidth,
    // centred on trackCenter. Depth & height stay fixed (from trackCenter).
    Vector3 SourcePosition(Vector3 actorCentroid)
    {
        Vector3 reference = stableTracking ? TrackingPoint(actorCentroid) : actorCentroid;
        float horiz = AxisValue(reference, stageHorizontalAxis);
        float n = Mathf.Clamp((horiz - horizontalCenter) / Mathf.Max(interactionWidth, 0.01f), -0.5f, 0.5f);
        Vector3 dir = trackAxis.sqrMagnitude > 1e-6f ? trackAxis.normalized : Vector3.right;
        return trackCenter + dir * (n * trackWidth);
    }

    // Robust horizontal reference: the average of the visible torso joints (neck +
    // hips). These barely move relative to the body, so the cloud slides smoothly
    // with the actor instead of jittering with every arm/leg swing. Falls back to
    // the full-body centroid when no torso joint is visible.
    Vector3 TrackingPoint(Vector3 fallback)
    {
        Vector3 sum = Vector3.zero;
        int c = 0;
        foreach (int idx in TORSO_KEYS)
            if (idx < MAX_ATTRACTORS && _jointValid[idx] && _jointSeen[idx]) { sum += _smoothed[idx]; c++; }
        return c > 0 ? sum / c : fallback;
    }

    static float AxisValue(Vector3 v, Axis a) => a == Axis.X ? v.x : (a == Axis.Y ? v.y : v.z);

    Vector3 ToUnity(float x, float y, float z)
    {
        Vector3 p = rosToUnity ? new Vector3(-y, z, x) : new Vector3(x, y, z);
        return p * scale + offset;
    }

    int Groups(int n) => Mathf.CeilToInt(n / 64f);

    void BuildRamp()
    {
        if (_ramp == null)
        {
            _ramp = new Texture2D(256, 1, TextureFormat.RGBA32, false)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        }
        var px = new Color[256];
        for (int i = 0; i < 256; i++) px[i] = colors.Evaluate(i / 255f);
        _ramp.SetPixels(px);
        _ramp.Apply();
    }

    void OnValidate()
    {
        if (Application.isPlaying && _ramp != null && colors != null) BuildRamp();
    }

    static Gradient DefaultGradient()
    {
        var g = new Gradient();
        g.SetKeys(
            new[] {
                new GradientColorKey(new Color(0.05f, 0.10f, 0.60f), 0.0f),
                new GradientColorKey(new Color(0.10f, 0.70f, 1.00f), 0.40f),
                new GradientColorKey(new Color(0.90f, 0.20f, 0.95f), 0.75f),
                new GradientColorKey(new Color(1.00f, 0.95f, 0.80f), 1.0f),
            },
            new[] {
                new GradientAlphaKey(0.55f, 0.0f),
                new GradientAlphaKey(1.00f, 1.0f),
            });
        return g;
    }

    void ReleaseBuffers()
    {
        _particles?.Release(); _particles = null;
        _attractors?.Release(); _attractors = null;
    }

    void ReleaseAll()
    {
        ReleaseBuffers();
        if (_mat != null) { Destroy(_mat); _mat = null; }
        if (_ramp != null) { Destroy(_ramp); _ramp = null; }
    }
}
