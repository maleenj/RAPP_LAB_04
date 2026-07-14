// ExampleJointBars.cs — a row of 3D bars that rise/fall with the joint angles.
//
// A second, even-simpler example (complements JointVisualizer's cube rotations).
// Reads a flat joint channel and scales one bar per joint.
//
// SETUP: add a EnactData component (channel = "robot_joint_states") and this script
// to an empty GameObject. Press Play.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class ExampleJointBars : MonoBehaviour
{
    public float spacing = 1.2f;
    public float heightScale = 0.5f;   // metres per radian
    [Range(0f, 0.95f)] public float smoothing = 0.4f;

    EnactData data;
    Transform[] bars;
    float[] current;

    void Awake() { data = GetComponent<EnactData>(); }

    void Update()
    {
        if (!data.HasData) return;
        float[] v = data.Values;
        if (v == null) return;

        if (bars == null || bars.Length != v.Length) Build(v.Length);

        for (int i = 0; i < v.Length; i++)
        {
            current[i] = Mathf.Lerp(v[i], current[i], smoothing);
            float h = Mathf.Max(0.02f, Mathf.Abs(current[i]) * heightScale);
            var b = bars[i];
            b.localScale = new Vector3(0.4f, h, 0.4f);
            b.localPosition = new Vector3(i * spacing, h * 0.5f, 0f);
            var rend = b.GetComponent<Renderer>();
            if (rend != null) rend.material.color = Color.HSVToRGB(Mathf.Clamp01(i / 6f), 0.7f, 1f);
        }
    }

    void Build(int n)
    {
        if (bars != null) foreach (var t in bars) if (t) Destroy(t.gameObject);
        bars = new Transform[n];
        current = new float[n];
        for (int i = 0; i < n; i++)
        {
            var c = GameObject.CreatePrimitive(PrimitiveType.Cube);
            c.name = $"bar_{i}";
            c.transform.SetParent(transform, false);
            bars[i] = c.transform;
        }
    }
}
