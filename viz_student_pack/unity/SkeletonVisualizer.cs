// SkeletonVisualizer.cs — draw the human skeleton in 3D.
//
// Reads the "skeleton" channel ([K,3] keypoints, K=18 for ZED BODY_18) and draws
// a sphere per joint plus connecting bones. An example to read and remix.
//
// SETUP:
//   1. Make sure the skeleton relay is running on the host:
//        ros2 run vam_inference skeleton_relay
//      and the bridge has the "skeleton" channel (it does, by default).
//   2. Add a EnactData component (set Channel = "skeleton") + this script to a
//      GameObject. Press Play. Add a FlyCamera to move around it.
//
// NOTE ON AXES: ROS uses X-forward, Y-left, Z-up; Unity uses X-right, Y-up,
// Z-forward. `rosToUnity` remaps for you. If your skeleton looks rotated/mirrored,
// toggle it or tweak `scale`/`offset` — coordinate conventions vary by capture.

using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class SkeletonVisualizer : MonoBehaviour
{
    [Header("Appearance")]
    public float jointScale = 0.06f;
    public float boneWidth = 0.02f;
    public Color jointColor = new Color(0.2f, 0.9f, 1f);
    public Color boneColor = new Color(1f, 1f, 1f, 0.9f);

    [Header("Coordinate mapping")]
    [Tooltip("Remap ROS (x fwd, y left, z up) -> Unity (x right, y up, z fwd).")]
    public bool rosToUnity = true;
    public float scale = 1f;
    public Vector3 offset = Vector3.zero;
    [Tooltip("Treat exact (0,0,0) keypoints as 'missing' and hide them.")]
    public bool skipZeroPoints = true;

    // ZED BODY_18 bone connections (pairs of keypoint indices).
    static readonly int[,] BODY_18_BONES = {
        {0,1},{1,2},{2,3},{3,4},{1,5},{5,6},{6,7},{1,8},{8,9},{9,10},
        {1,11},{11,12},{12,13},{0,14},{14,16},{0,15},{15,17}
    };

    EnactData data;
    Transform[] joints;
    LineRenderer[] bones;
    Material _mat;

    void Awake()
    {
        data = GetComponent<EnactData>();
        _mat = new Material(Shader.Find("Sprites/Default"));
    }

    void Update()
    {
        if (!data.HasData) return;
        var f = data.Frame;
        int k = f.Rows;                 // [K,3]
        if (k <= 0 || f.Cols < 3) return;

        EnsurePools(k);

        // place joints
        var world = new Vector3[k];
        var valid = new bool[k];
        for (int i = 0; i < k; i++)
        {
            float x = f.Get(i, 0), y = f.Get(i, 1), z = f.Get(i, 2);
            bool isZero = (x == 0f && y == 0f && z == 0f);
            valid[i] = !(skipZeroPoints && isZero);
            world[i] = ToUnity(x, y, z);
            joints[i].gameObject.SetActive(valid[i]);
            if (valid[i])
            {
                joints[i].position = world[i];
                joints[i].localScale = Vector3.one * jointScale;
            }
        }

        // draw bones
        for (int b = 0; b < bones.Length; b++)
        {
            int a = BODY_18_BONES[b, 0], c = BODY_18_BONES[b, 1];
            bool show = a < k && c < k && valid[a] && valid[c];
            bones[b].enabled = show;
            if (show)
            {
                bones[b].SetPosition(0, world[a]);
                bones[b].SetPosition(1, world[c]);
            }
        }
    }

    Vector3 ToUnity(float x, float y, float z)
    {
        Vector3 p = rosToUnity ? new Vector3(-y, z, x) : new Vector3(x, y, z);
        return p * scale + offset;
    }

    void EnsurePools(int k)
    {
        if (joints != null && joints.Length == k) return;

        if (joints != null) foreach (var t in joints) if (t) Destroy(t.gameObject);
        if (bones != null) foreach (var lr in bones) if (lr) Destroy(lr.gameObject);

        joints = new Transform[k];
        for (int i = 0; i < k; i++)
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            s.name = $"joint_{i}";
            s.transform.SetParent(transform, false);
            var r = s.GetComponent<Renderer>();
            r.material = _mat;
            r.material.color = jointColor;
            joints[i] = s.transform;
        }

        int nb = BODY_18_BONES.GetLength(0);
        bones = new LineRenderer[nb];
        for (int b = 0; b < nb; b++)
        {
            var go = new GameObject($"bone_{b}");
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.material = _mat;
            lr.startColor = lr.endColor = boneColor;
            lr.startWidth = lr.endWidth = boneWidth;
            lr.positionCount = 2;
            lr.useWorldSpace = true;
            bones[b] = lr;
        }
    }
}
