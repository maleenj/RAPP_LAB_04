// EnactInspector.cs — the default "just show me everything" visualizer.
//
// Attach this to ONE GameObject (alongside or near a EnactClient) and press Play.
// It draws an on-screen panel listing EVERY channel currently arriving:
//   * flat vectors  -> labeled value + bar
//   * matrices/tensors -> a colored heatmap grid (attention, etc.)
// Zero scene setup. Use it to confirm data is flowing and to get a feel for what
// each channel looks like before you build your own visuals.

using System.Collections.Generic;
using UnityEngine;

public class EnactInspector : MonoBehaviour
{
    [Tooltip("Leave empty to auto-find the scene's EnactClient.")]
    public EnactClient client;

    [Header("Layout")]
    public float panelWidth = 420f;
    public int cellPixels = 16;        // heatmap cell size
    public int maxBarsPerChannel = 12; // cap long vectors in the overlay

    Texture2D _white;
    Vector2 _scroll;

    void Awake()
    {
        _white = new Texture2D(1, 1);
        _white.SetPixel(0, 0, Color.white);
        _white.Apply();
    }

    void Start()
    {
        if (client == null) client = EnactClient.Instance != null ? EnactClient.Instance : FindObjectOfType<EnactClient>();
    }

    void OnGUI()
    {
        if (client == null) { GUI.Label(new Rect(12, 12, 400, 24), "EnactInspector: no EnactClient found."); return; }

        float x = 12f, y = 12f;
        GUI.color = Color.white;
        GUI.Box(new Rect(x, y, panelWidth, 28), "");
        GUI.Label(new Rect(x + 8, y + 5, panelWidth, 22),
            client.IsConnected ? $"ENACT ● connected   {client.Hz:F0} Hz   channels: {client.LatestByChannel.Count}"
                               : "ENACT ○ connecting…");
        y += 34f;

        float viewHeight = Screen.height - y - 12f;
        var keys = new List<string>(client.LatestByChannel.Keys);
        keys.Sort();

        _scroll = GUI.BeginScrollView(new Rect(x, y, panelWidth + 20, viewHeight),
                                      _scroll, new Rect(0, 0, panelWidth, EstimateHeight(keys)));
        float cy = 0f;
        foreach (var ch in keys)
        {
            var f = client.LatestByChannel[ch];
            cy = DrawChannel(ch, f, 0f, cy);
            cy += 10f;
        }
        GUI.EndScrollView();
    }

    float EstimateHeight(List<string> keys)
    {
        float h = 0f;
        foreach (var ch in keys)
        {
            var f = client.LatestByChannel[ch];
            h += 24f;
            if (f.HasTensors) foreach (var name in f.tensors.Keys) h += TensorHeight(f.tensors[name]);
            else if (f.IsMatrix) h += f.Rows * cellPixels + 18f;
            else h += Mathf.Min(f.Length, maxBarsPerChannel) * 16f + 4f;
            h += 10f;
        }
        return h + 20f;
    }

    float TensorHeight(EnactTensor t)
    {
        if (t.shape != null && t.shape.Length >= 2) return t.Rows * cellPixels + 18f;
        return Mathf.Min(t.Length, maxBarsPerChannel) * 16f + 18f;
    }

    float DrawChannel(string ch, EnactFrame f, float x, float y)
    {
        GUI.color = new Color(0.6f, 0.8f, 1f);
        GUI.Label(new Rect(x, y, panelWidth, 20), $"▸ {ch}");
        GUI.color = Color.white;
        y += 22f;

        if (f.HasTensors)
        {
            foreach (var name in f.tensors.Keys)
            {
                var t = f.tensors[name];
                GUI.Label(new Rect(x + 8, y, panelWidth, 18), $"{name} [{string.Join(",", t.shape)}]");
                y += 18f;
                if (t.shape != null && t.shape.Length >= 2)
                    y = DrawHeatmap(t.MeanPlane(), t.Rows, t.Cols, x + 8, y);
                else
                    y = DrawBars(t.data, null, x + 8, y);
            }
        }
        else if (f.IsMatrix)
        {
            y = DrawHeatmap(f.data, f.Rows, f.Cols, x + 8, y);
        }
        else
        {
            y = DrawBars(f.data, f.labels, x + 8, y);
        }
        return y;
    }

    float DrawBars(float[] data, string[] labels, float x, float y)
    {
        if (data == null) return y;
        float maxAbs = 1e-6f;
        foreach (var v in data) maxAbs = Mathf.Max(maxAbs, Mathf.Abs(v));
        int n = Mathf.Min(data.Length, maxBarsPerChannel);
        for (int i = 0; i < n; i++)
        {
            string lbl = (labels != null && i < labels.Length) ? labels[i] : i.ToString();
            GUI.Label(new Rect(x, y, 150, 16), $"{lbl}: {data[i]:F3}");
            float w = (Mathf.Abs(data[i]) / maxAbs) * 180f;
            GUI.color = new Color(0.2f, 0.85f, 0.5f);
            GUI.DrawTexture(new Rect(x + 160, y + 2, Mathf.Max(1f, w), 11), _white);
            GUI.color = Color.white;
            y += 16f;
        }
        if (data.Length > n) { GUI.Label(new Rect(x, y, 200, 16), $"… +{data.Length - n} more"); y += 16f; }
        return y + 4f;
    }

    float DrawHeatmap(float[] cell, int rows, int cols, float x, float y)
    {
        if (cell == null || rows <= 0 || cols <= 0) return y;
        float max = 1e-6f;
        foreach (var v in cell) max = Mathf.Max(max, Mathf.Abs(v));
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                float t = Mathf.Clamp01(Mathf.Abs(cell[r * cols + c]) / max);
                GUI.color = new Color(t, 0.25f * t, 1f - t);   // blue -> red
                GUI.DrawTexture(new Rect(x + c * cellPixels, y + r * cellPixels,
                                         cellPixels - 1, cellPixels - 1), _white);
            }
        GUI.color = Color.white;
        return y + rows * cellPixels + 4f;
    }
}
