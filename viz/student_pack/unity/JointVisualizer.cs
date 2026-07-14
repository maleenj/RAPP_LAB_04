// JointVisualizer.cs — drive a row of cubes from the joint-angle stream.
//
// The simplest possible "it works" visual: each incoming joint angle rotates one
// cube. Assign 6 cube Transforms (or leave empty to auto-spawn them). Point the
// `client` field at your EnactClient and pick which channel to read.
//
// This is a starting point — replace the cubes with your own rig / robot model
// and map `data[i]` however you like. The data is joint angles in RADIANS.

using UnityEngine;

public class JointVisualizer : MonoBehaviour
{
    [Tooltip("The EnactClient providing frames.")]
    public EnactClient client;

    [Tooltip("Which joint channel to visualize (the robot's actual joints).")]
    public string channel = "robot_joint_states";

    [Tooltip("Cubes to rotate. Leave empty to auto-create a row of 6.")]
    public Transform[] joints;

    [Tooltip("Local axis each cube rotates about.")]
    public Vector3 rotationAxis = Vector3.up;

    [Tooltip("Degrees per radian of incoming angle (visual gain).")]
    public float degreesPerRadian = Mathf.Rad2Deg;

    [Tooltip("Smoothing (0 = snap, 0.9 = very smooth).")]
    [Range(0f, 0.98f)] public float smoothing = 0.5f;

    [Tooltip("Side length of each auto-spawned cube, in metres (0.25 = 25 cm).")]
    public float cubeSize = 0.25f;

    float[] _current;

    void Start()
    {
        if (joints == null || joints.Length == 0) AutoSpawn(6);
        _current = new float[joints.Length];
        if (client == null) client = FindObjectOfType<EnactClient>();
    }

    void AutoSpawn(int n)
    {
        joints = new Transform[n];
        for (int i = 0; i < n; i++)
        {
            var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = $"Joint_{i}";
            cube.transform.SetParent(transform, false);
            cube.transform.localScale = Vector3.one * cubeSize;   // 25 cm cube
            cube.transform.localPosition = new Vector3(i * cubeSize * 1.5f, 0, 0);
            joints[i] = cube.transform;
        }
    }

    void Update()
    {
        if (client == null) return;
        if (!client.LatestByChannel.TryGetValue(channel, out var frame)) return;
        if (frame.data == null) return;

        int n = Mathf.Min(joints.Length, frame.data.Length);
        for (int i = 0; i < n; i++)
        {
            float target = frame.data[i];
            // exponential smoothing toward the latest target
            _current[i] = Mathf.Lerp(target, _current[i], smoothing);
            float deg = _current[i] * degreesPerRadian;
            if (joints[i] != null)
                joints[i].localRotation = Quaternion.AngleAxis(deg, rotationAxis);
        }
    }
}
