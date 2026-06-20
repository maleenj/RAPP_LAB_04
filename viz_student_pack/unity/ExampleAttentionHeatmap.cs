// ExampleAttentionHeatmap.cs — a "brain scan" wall in 3D.
//
// Builds a grid of quads in world space and colors each cell by the model's
// attention. Reads a tensor channel (default "activations" -> "decoder_xattn",
// [2,4,10,10]) and averages it down to one 10x10 map.
//
// SETUP: add a EnactData component (channel = "activations") and this script to an
// empty GameObject. Point your camera at it. Press Play (live robot, or a
// recording that contains activations).
//
// This is an EXAMPLE to read and remix — not the only way to do it.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class ExampleAttentionHeatmap : MonoBehaviour
{
    [Tooltip("Which tensor inside the channel to draw (averaged over its layers/heads).")]
    public string tensorName = "decoder_xattn";

    [Tooltip("World size of one cell.")]
    public float cellSize = 0.12f;

    public Gradient colors = DefaultGradient();

    EnactData data;
    Transform[,] cells;
    int rows, cols;

    void Awake() { data = GetComponent<EnactData>(); }

    void Update()
    {
        if (!data.HasData) return;
        EnactTensor t = data.Tensor(tensorName);
        if (t == null || t.Rows <= 0 || t.Cols <= 0) return;

        if (cells == null || rows != t.Rows || cols != t.Cols)
            BuildGrid(t.Rows, t.Cols);

        float[] map = t.MeanPlane();           // [rows*cols], collapsed attention
        float max = 1e-6f;
        foreach (var v in map) max = Mathf.Max(max, Mathf.Abs(v));

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                float w = Mathf.Clamp01(Mathf.Abs(map[r * cols + c]) / max);
                var cell = cells[r, c];
                var rend = cell.GetComponent<Renderer>();
                if (rend != null) rend.material.color = colors.Evaluate(w);
                // also pop the cell outward a touch with its weight
                var p = cell.localPosition;
                cell.localPosition = new Vector3(p.x, p.y, -w * cellSize * 1.5f);
            }
    }

    void BuildGrid(int r, int c)
    {
        if (cells != null)
            foreach (var t in cells) if (t) Destroy(t.gameObject);

        rows = r; cols = c;
        cells = new Transform[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
            {
                var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
                q.name = $"cell_{i}_{j}";
                q.transform.SetParent(transform, false);
                q.transform.localScale = Vector3.one * (cellSize * 0.92f);
                // center the grid on this object
                float ox = (cols - 1) * 0.5f, oy = (rows - 1) * 0.5f;
                q.transform.localPosition = new Vector3((j - ox) * cellSize, (oy - i) * cellSize, 0f);
                cells[i, j] = q.transform;
            }
    }

    static Gradient DefaultGradient()
    {
        var g = new Gradient();
        g.SetKeys(
            new[] { new GradientColorKey(new Color(0.05f, 0.1f, 0.4f), 0f),
                    new GradientColorKey(new Color(1f, 0.85f, 0.1f), 1f) },
            new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(1f, 1f) });
        return g;
    }
}
