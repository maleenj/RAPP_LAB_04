// RadialMandala.cs — Example 02 (special): a kinetic geometric sculpture driven by
// the 6 robot joint angles (radians). Six concentric ring (torus) meshes are
// staggered in depth to form a tunnel you look into; each ring tumbles about its
// own axis, rotated DIRECTLY by its joint angle. Crisp single-colour emissive
// materials with an edge-glow rim — a clean geometric counterpoint to the organic
// particle mode. When an angle crosses a threshold, its ring snaps a bright flash.
//
// CHANNEL: "robot_joint_states" (a flat [6] vector, radians).
//
// SETUP:
//   1. Create an empty GameObject → Add Component → RadialMandala.
//      (An EnactData is added automatically.)
//   2. On the EnactData component set Channel = "robot_joint_states".
//   3. Assign Render Shader -> the Enact/RadialMandala shader (the .shader file).
//   4. Make sure an EnactClient exists and is receiving data. Press Play.
//      A dark background + HDR/Bloom makes the edge-glow pop.
//
// This is Option A from the brief (nested rings). For Option B (radial bars) you'd
// swap BuildTorus() for a flat bar/wedge mesh and arrange them around a circle —
// the data plumbing and flash logic below stay the same.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class RadialMandala : MonoBehaviour
{
    [Header("Assets (assign this)")]
    [Tooltip("The Enact/RadialMandala shader (the .shader file in this folder).")]
    public Shader renderShader;

    [Header("Geometry — the tunnel of rings")]
    [Tooltip("Major radius of the innermost ring, in metres.")]
    public float baseRadius = 0.6f;
    [Tooltip("How much larger each successive ring is (concentric tunnel).")]
    public float radiusStep = 0.35f;
    [Tooltip("Tube thickness of each ring — small = thin, crisp rings.")]
    public float tubeRadius = 0.05f;
    [Tooltip("How far apart the rings sit along Z (the depth of the tunnel).")]
    public float depthStep = 0.5f;
    [Tooltip("Smoothness of each ring: segments around the major circle.")]
    [Range(12, 256)] public int ringSegments = 96;
    [Tooltip("Smoothness of the tube cross-section.")]
    [Range(4, 48)] public int tubeSegments = 16;

    [Header("Motion — angle → rotation")]
    [Tooltip("Degrees of tumble per radian of joint angle (visual gain).")]
    public float degreesPerRadian = Mathf.Rad2Deg;
    [Tooltip("Smoothing toward the latest angle (0 = snap, 0.95 = very smooth).")]
    [Range(0f, 0.97f)] public float smoothing = 0.5f;
    [Tooltip("Ambient idle spin (deg/sec) added on top, so it breathes when the robot is still. 0 = pure data.")]
    public float ambientSpin = 0f;

    [Header("Look — colour & glow")]
    [Tooltip("Ring colour across the set (inner → outer). HDR-friendly with Bloom.")]
    public Gradient colors;
    [Tooltip("Base emissive brightness (push >1 for Bloom).")]
    public float intensity = 1.5f;
    [Tooltip("Strength of the fresnel rim / edge glow on each ring.")]
    [Range(0f, 8f)] public float edgeGlow = 2.5f;

    [Header("Accent flashes — snap when an angle crosses a threshold")]
    [Tooltip("Fire a flash each time a joint angle crosses a multiple of this many radians. 0 = off.")]
    public float flashEvery = 0.5f;
    [Tooltip("How bright the flash spikes (added emissive).")]
    public float flashStrength = 6f;
    [Tooltip("How fast a flash fades back (per second).")]
    public float flashDecay = 6f;

    [Header("Camera — optional slow orbit for parallax")]
    [Tooltip("Assign a camera Transform to orbit it gently around the tunnel mouth. Leave empty to fly yourself.")]
    public Transform orbitCamera;
    [Tooltip("Orbit speed, degrees/sec.")]
    public float orbitSpeed = 8f;
    [Tooltip("Distance the camera sits in front of the tunnel mouth.")]
    public float orbitDistance = 4f;
    [Tooltip("Vertical tilt of the orbit, degrees (look slightly down/up into the tunnel).")]
    public float orbitTilt = 10f;

    // Six distinct tumble axes so the rings never move in lockstep — this is what
    // makes them read as interlocking moving parts rather than one spinning disc.
    static readonly Vector3[] AXES =
    {
        Vector3.right,
        Vector3.up,
        new Vector3(1f, 1f, 0f),
        new Vector3(0f, 1f, 1f),
        new Vector3(1f, 0f, 1f),
        new Vector3(1f, 1f, 1f),
    };

    EnactData _data;
    Transform[] _rings;
    Material[] _mats;
    float[] _current;     // smoothed angle per ring
    float[] _flash;       // current flash brightness per ring
    int _built;           // how many rings currently exist
    float _orbitAngle;

    void Reset() { colors = DefaultGradient(); }

    void OnEnable()
    {
        _data = GetComponent<EnactData>();
        if (colors == null || colors.colorKeys.Length == 0) colors = DefaultGradient();
        if (renderShader == null)
        {
            Debug.LogError("[RadialMandala] Assign the Enact/RadialMandala Render Shader in the Inspector.");
            enabled = false;
        }
    }

    void OnDisable() => Clear();

    void Update()
    {
        if (!_data.HasData) return;
        float[] v = _data.Values;
        if (v == null || v.Length == 0) return;

        int n = v.Length;
        if (_built != n) Build(n);

        float dt = Time.deltaTime;
        float ambient = ambientSpin * Time.time;

        for (int i = 0; i < n; i++)
        {
            // smooth toward the latest angle (same convention as the other examples)
            float prev = _current[i];
            _current[i] = Mathf.Lerp(v[i], _current[i], smoothing);

            // accent flash: did we cross a multiple of flashEvery this frame?
            if (flashEvery > 1e-4f &&
                Mathf.FloorToInt(prev / flashEvery) != Mathf.FloorToInt(_current[i] / flashEvery))
                _flash[i] = flashStrength;
            _flash[i] = Mathf.Max(0f, _flash[i] - flashDecay * dt);

            // direct mapping: angle (deg) → tumble about this ring's own axis
            float deg = _current[i] * degreesPerRadian + ambient;
            _rings[i].localRotation = Quaternion.AngleAxis(deg, AXES[i % AXES.Length].normalized);

            _mats[i].SetFloat("_Intensity", intensity);
            _mats[i].SetFloat("_EdgeGlow", edgeGlow);
            _mats[i].SetFloat("_Flash", _flash[i]);
        }

        if (orbitCamera != null) OrbitCamera(dt);
    }

    void OrbitCamera(float dt)
    {
        _orbitAngle += orbitSpeed * dt;
        Vector3 center = transform.position;
        // Sit in front of the tunnel mouth (-Z) and circle around it.
        Vector3 offset = Quaternion.Euler(orbitTilt, _orbitAngle, 0f) * (Vector3.back * orbitDistance);
        orbitCamera.position = center + offset;
        orbitCamera.LookAt(center);
    }

    void Build(int n)
    {
        Clear();
        _rings = new Transform[n];
        _mats = new Material[n];
        _current = new float[n];
        _flash = new float[n];

        // Centre the tunnel on the GameObject: rings span symmetrically along Z.
        float z0 = -0.5f * (n - 1) * depthStep;

        for (int i = 0; i < n; i++)
        {
            var go = new GameObject($"ring_{i}");
            go.transform.SetParent(transform, false);
            go.transform.localPosition = new Vector3(0f, 0f, z0 + i * depthStep);

            var mf = go.AddComponent<MeshFilter>();
            mf.sharedMesh = BuildTorus(baseRadius + i * radiusStep, tubeRadius, ringSegments, tubeSegments);

            var mr = go.AddComponent<MeshRenderer>();
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            mr.receiveShadows = false;
            _mats[i] = new Material(renderShader);
            Color c = colors.Evaluate(n > 1 ? i / (float)(n - 1) : 0f);
            _mats[i].SetColor("_Color", c);
            mr.material = _mats[i];

            _rings[i] = go.transform;
        }
        _built = n;
    }

    // A torus (ring) mesh in the XY plane (normal along Z), so a row of them along
    // Z reads as a tunnel. major = ring radius, minor = tube thickness.
    static Mesh BuildTorus(float major, float minor, int seg, int tubeSeg)
    {
        seg = Mathf.Max(3, seg);
        tubeSeg = Mathf.Max(3, tubeSeg);
        int vcount = (seg + 1) * (tubeSeg + 1);
        var verts = new Vector3[vcount];
        var norms = new Vector3[vcount];
        var tris = new int[seg * tubeSeg * 6];

        int vi = 0;
        for (int s = 0; s <= seg; s++)
        {
            float u = s / (float)seg * Mathf.PI * 2f;
            Vector3 center = new Vector3(Mathf.Cos(u) * major, Mathf.Sin(u) * major, 0f);
            Vector3 outDir = new Vector3(Mathf.Cos(u), Mathf.Sin(u), 0f);   // radial, in-plane
            for (int t = 0; t <= tubeSeg; t++, vi++)
            {
                float w = t / (float)tubeSeg * Mathf.PI * 2f;
                // tube cross-section spans the radial (outDir) and Z directions
                Vector3 normal = outDir * Mathf.Cos(w) + Vector3.forward * Mathf.Sin(w);
                verts[vi] = center + normal * minor;
                norms[vi] = normal;
            }
        }

        int ti = 0, ring = tubeSeg + 1;
        for (int s = 0; s < seg; s++)
            for (int t = 0; t < tubeSeg; t++)
            {
                int a = s * ring + t;
                int b = a + ring;
                tris[ti++] = a;     tris[ti++] = b;     tris[ti++] = a + 1;
                tris[ti++] = a + 1; tris[ti++] = b;     tris[ti++] = b + 1;
            }

        var m = new Mesh { name = "Torus" };
        if (vcount > 65535) m.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        m.vertices = verts;
        m.normals = norms;
        m.triangles = tris;
        m.RecalculateBounds();
        return m;
    }

    void OnValidate()
    {
        // Rebuild geometry if the tunnel shape changed while playing.
        if (Application.isPlaying && _built > 0) _built = -1;
    }

    static Gradient DefaultGradient()
    {
        var g = new Gradient();
        g.SetKeys(
            new[]
            {
                new GradientColorKey(new Color(0.10f, 0.70f, 1.00f), 0.0f),
                new GradientColorKey(new Color(0.55f, 0.30f, 1.00f), 0.5f),
                new GradientColorKey(new Color(1.00f, 0.35f, 0.75f), 1.0f),
            },
            new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(1f, 1f) });
        return g;
    }

    void Clear()
    {
        if (_rings != null)
            foreach (var t in _rings) if (t) Destroy(t.gameObject);
        if (_mats != null)
            foreach (var m in _mats) if (m) Destroy(m);
        _rings = null; _mats = null; _current = null; _flash = null; _built = 0;
    }
}
