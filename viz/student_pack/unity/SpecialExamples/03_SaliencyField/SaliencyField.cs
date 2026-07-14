// SaliencyField.cs — Example 03 (special): the input-saliency map blown up to FILL
// THE SCREEN as a living heat-field. Reads the saliency tensor (activations ->
// input_saliency, [10 timesteps x K keypoints]) and draws one quad per cell. Each
// cell is coloured by a modern heat gradient AND, driven by the same value, it
// tilts and pushes toward the camera — so the flat heatmap becomes a rippling 3D
// surface. A built-in camera auto-fit frames the whole grid edge-to-edge.
//
// CHANNEL: "activations"  (tensor "input_saliency").
//
// SETUP:
//   1. New scene → empty GameObject → Add Component → SaliencyField.
//      (An EnactData is added automatically.)
//   2. On the EnactData component set Channel = "activations".
//   3. Assign Cell Shader -> the Enact/SaliencyField shader (the .shader file).
//   4. Leave "Fit Camera" ON and press Play — the camera frames the grid for you.
//      Dark background + HDR/Bloom makes the heat glow.
//
// The depth push is deliberately gentle (see Depth Scale): cells lift toward you
// but stay knit into the surface so the grid reads as one cohesive sheet, not
// scattered tiles. Crank the knobs to taste.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class SaliencyField : MonoBehaviour
{
    [Header("Assets (assign this)")]
    [Tooltip("The Enact/SaliencyField shader (the .shader file in this folder).")]
    public Shader cellShader;

    [Header("Data")]
    [Tooltip("Which tensor inside the 'activations' channel to draw.")]
    public string tensorName = "input_saliency";
    [Tooltip("Average the 10 timesteps into a single row (one cell per keypoint).")]
    public bool collapseTimesteps = false;

    [Header("Grid layout")]
    [Tooltip("Centre-to-centre spacing of cells, in metres.")]
    public float cellSize = 0.3f;
    [Tooltip("Gap between cells as a fraction of cell size (0 = touching).")]
    [Range(0f, 0.6f)] public float cellGap = 0.1f;

    [Header("Colour — the heatmap")]
    [Tooltip("Low → high value colour ramp (Magma by default).")]
    public Gradient colors;
    [Tooltip("Contrast shaping: >1 pushes mid values dark, <1 lifts them.")]
    [Range(0.2f, 4f)] public float colorGamma = 1f;
    [Tooltip("Emissive brightness — push >1 and add Bloom for the glow.")]
    public float intensity = 1.4f;

    [Header("Normalisation (value → 0..1)")]
    [Tooltip("ON: scale by the brightest cell each frame. OFF: divide by Manual Max.")]
    public bool autoNormalize = true;
    [Tooltip("Used when Auto Normalize is OFF.")]
    public float manualMax = 1f;
    [Tooltip("Smooths the running max so the field doesn't flicker (0 = instant).")]
    [Range(0f, 0.99f)] public float maxSmoothing = 0.9f;

    [Header("Motion — the same value drives these")]
    [Tooltip("Per-cell easing toward the latest value (0 = snap, 0.95 = very smooth).")]
    [Range(0f, 0.97f)] public float valueSmoothing = 0.6f;
    [Tooltip("Degrees a cell tilts at value = 1.")]
    public float rotationMax = 35f;
    [Tooltip("Local axis each cell tilts about (X = flip toward you, Y = swing sideways).")]
    public Vector3 rotationAxis = new Vector3(1f, 0f, 0f);
    [Tooltip("Extra in-plane spin (about Z) at value = 1. 0 = off.")]
    public float spinMax = 0f;
    [Tooltip("Metres a cell lifts toward the camera at value = 1. Keep SMALL so the " +
             "grid stays a cohesive sheet (the depth is meant to be felt, not flown).")]
    public float depthScale = 0.25f;

    [Header("Camera auto-fit (full screen)")]
    [Tooltip("ON: drive a camera to frame the whole grid edge-to-edge each frame.")]
    public bool fitCamera = true;
    [Tooltip("Camera to drive (leave empty = Camera.main).")]
    public Camera targetCamera;
    [Tooltip("How much of the screen the grid fills (0.9 = small margin).")]
    [Range(0.3f, 1f)] public float fillFraction = 0.9f;
    [Tooltip("Tilt the view by this many degrees so the depth motion reads as 3D. " +
             "0 = dead-on / flattest full-screen look.")]
    public float viewTilt = 8f;

    EnactData _data;
    Transform[,] _cells;
    MeshRenderer[,] _rend;
    MaterialPropertyBlock _mpb;
    Material _mat;
    float[,] _val;            // smoothed 0..1 value per cell
    Vector3[,] _basePos;      // resting position of each cell (before depth lift)
    int _rows, _cols;
    float _runningMax = 1e-3f;
    int _colorId;

    void Reset() { colors = MagmaGradient(); }

    void OnEnable()
    {
        _data = GetComponent<EnactData>();
        if (colors == null || colors.colorKeys.Length == 0) colors = MagmaGradient();
        if (cellShader == null)
        {
            Debug.LogError("[SaliencyField] Assign the Enact/SaliencyField Cell Shader in the Inspector.");
            enabled = false;
            return;
        }
        _mat = new Material(cellShader);
        _mpb = new MaterialPropertyBlock();
        _colorId = Shader.PropertyToID("_Color");
    }

    void OnDisable() => Clear();

    void Update()
    {
        if (!_data.HasData) return;
        EnactTensor t = _data.Tensor(tensorName);
        if (t == null || t.Rows <= 0 || t.Cols <= 0) return;

        int wantRows = collapseTimesteps ? 1 : t.Rows;
        int wantCols = t.Cols;
        if (_cells == null || _rows != wantRows || _cols != wantCols)
            BuildGrid(wantRows, wantCols);

        // --- gather the values (abs, optionally collapsed over timesteps) ---
        float frameMax = 1e-6f;
        for (int r = 0; r < _rows; r++)
            for (int c = 0; c < _cols; c++)
                frameMax = Mathf.Max(frameMax, Mathf.Abs(SampleRaw(t, r, c)));

        _runningMax = Mathf.Max(1e-5f, Mathf.Lerp(frameMax, _runningMax, maxSmoothing));
        float denom = autoNormalize ? _runningMax : Mathf.Max(manualMax, 1e-5f);

        _mat.SetFloat("_Intensity", intensity);

        for (int r = 0; r < _rows; r++)
            for (int c = 0; c < _cols; c++)
            {
                float raw = Mathf.Clamp01(Mathf.Abs(SampleRaw(t, r, c)) / denom);
                float target = Mathf.Pow(raw, colorGamma);
                float v = Mathf.Lerp(target, _val[r, c], valueSmoothing);
                _val[r, c] = v;

                var cell = _cells[r, c];

                // colour (per-cell, via property block — no 180 material instances)
                _rend[r, c].GetPropertyBlock(_mpb);
                _mpb.SetColor(_colorId, colors.Evaluate(v));
                _rend[r, c].SetPropertyBlock(_mpb);

                // tilt + optional spin, both scaled by the value
                Quaternion rot = Quaternion.AngleAxis(v * rotationMax, SafeAxis(rotationAxis))
                               * Quaternion.AngleAxis(v * spinMax, Vector3.forward);
                cell.localRotation = rot;

                // gentle lift toward the camera (-Z, where the camera sits)
                Vector3 bp = _basePos[r, c];
                cell.localPosition = new Vector3(bp.x, bp.y, -v * depthScale);
            }

        if (fitCamera) FitCamera();
    }

    float SampleRaw(EnactTensor t, int r, int c)
    {
        if (!collapseTimesteps) return t.Get(r, c);
        // collapse: average this keypoint over all timesteps
        float s = 0f;
        for (int k = 0; k < t.Rows; k++) s += t.Get(k, c);
        return s / Mathf.Max(t.Rows, 1);
    }

    void BuildGrid(int rows, int cols)
    {
        Clear();
        _rows = rows; _cols = cols;
        _cells = new Transform[rows, cols];
        _rend = new MeshRenderer[rows, cols];
        _val = new float[rows, cols];
        _basePos = new Vector3[rows, cols];

        _mat.SetFloat("_Intensity", intensity);
        float quad = cellSize * (1f - cellGap);
        float ox = (cols - 1) * 0.5f, oy = (rows - 1) * 0.5f;

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
                q.name = $"cell_{r}_{c}";
                q.transform.SetParent(transform, false);
                var col = q.GetComponent<Collider>();
                if (col) Destroy(col);

                var mr = q.GetComponent<MeshRenderer>();
                mr.sharedMaterial = _mat;
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                mr.receiveShadows = false;

                q.transform.localScale = Vector3.one * quad;
                Vector3 p = new Vector3((c - ox) * cellSize, (oy - r) * cellSize, 0f);
                q.transform.localPosition = p;

                _basePos[r, c] = p;
                _cells[r, c] = q.transform;
                _rend[r, c] = mr;
            }
    }

    // Frame the whole grid edge-to-edge for the current screen aspect, then tilt
    // the view a touch so the depth/rotation motion reads as real 3D.
    void FitCamera()
    {
        var cam = targetCamera != null ? targetCamera : Camera.main;
        if (cam == null) return;

        float W = _cols * cellSize, H = _rows * cellSize;
        float vfov = cam.fieldOfView * Mathf.Deg2Rad;
        float distH = (H * 0.5f) / Mathf.Tan(vfov * 0.5f);
        float hfov = 2f * Mathf.Atan(Mathf.Tan(vfov * 0.5f) * Mathf.Max(cam.aspect, 1e-3f));
        float distW = (W * 0.5f) / Mathf.Tan(hfov * 0.5f);
        float dist = Mathf.Max(distH, distW) / Mathf.Max(fillFraction, 0.05f);

        Vector3 center = transform.position;
        Vector3 dir = Quaternion.Euler(viewTilt, 0f, 0f) * Vector3.back;  // camera sits in front (-Z)
        cam.transform.position = center + dir * dist;
        cam.transform.LookAt(center, Vector3.up);
    }

    static Vector3 SafeAxis(Vector3 a) => a.sqrMagnitude > 1e-6f ? a.normalized : Vector3.right;

    void OnValidate()
    {
        // Rebuild when layout-affecting fields change while playing.
        if (Application.isPlaying && _cells != null) { _rows = -1; }
    }

    // Magma: a modern, dramatic heat ramp (matplotlib-style).
    static Gradient MagmaGradient()
    {
        var g = new Gradient();
        g.SetKeys(
            new[]
            {
                new GradientColorKey(new Color(0.043f, 0.000f, 0.078f), 0.00f),
                new GradientColorKey(new Color(0.231f, 0.059f, 0.439f), 0.25f),
                new GradientColorKey(new Color(0.549f, 0.161f, 0.506f), 0.50f),
                new GradientColorKey(new Color(0.871f, 0.286f, 0.408f), 0.70f),
                new GradientColorKey(new Color(0.996f, 0.624f, 0.427f), 0.85f),
                new GradientColorKey(new Color(0.988f, 0.992f, 0.749f), 1.00f),
            },
            new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(1f, 1f) });
        return g;
    }

    void Clear()
    {
        if (_cells != null)
            foreach (var t in _cells) if (t) Destroy(t.gameObject);
        _cells = null; _rend = null; _val = null; _basePos = null; _rows = 0; _cols = 0;
    }
}
