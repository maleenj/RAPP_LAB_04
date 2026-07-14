// EnactVisualizerTemplate.cs — COPY ME to start a new visualization.
//
// This is your starting point. The data plumbing is done; you just write the
// visual. Steps:
//   1. Duplicate this file and rename the class (e.g. MyCoolViz) and the file.
//   2. Put it on a GameObject. A EnactData component is added automatically.
//   3. On that EnactData, set `channel` in the Inspector (see EnactData.cs for the list).
//   4. Fill in the marked block below with your visualization.
//
// You get the data three ways — use whichever you like:
//   * data.Values            -> float[] for flat channels (joints)
//   * data.Get(row, col)     -> a matrix element
//   * data.Tensor("name")    -> a EnactTensor for activation channels
// Joint values are in radians.

using UnityEngine;

[RequireComponent(typeof(EnactData))]
public class EnactVisualizerTemplate : MonoBehaviour
{
    EnactData data;

    void Awake()
    {
        data = GetComponent<EnactData>();
        // Prefer the event if you only want to react when fresh data arrives:
        data.OnData += OnNewFrame;
    }

    void OnNewFrame(EnactFrame frame)
    {
        // Called once per incoming frame for this channel. You can build your
        // visual here, or do it in Update() by polling — your choice.
    }

    void Update()
    {
        if (!data.HasData) return;

        // =====================================================================
        // ===============  YOUR VISUALIZATION CODE GOES HERE  =================
        // =====================================================================
        //
        // --- Example A: a flat vector channel (e.g. "robot_joint_states") ---
        //   float[] joints = data.Values;          // length 6, radians
        //   for (int i = 0; i < joints.Length; i++) {
        //       // map joints[i] to position / rotation / color / size ...
        //   }
        //
        // --- Example B: a matrix from an activation channel ("activations") ---
        //   EnactTensor attn = data.Tensor("decoder_xattn");   // [2,4,10,10]
        //   if (attn != null) {
        //       float[] map = attn.MeanPlane();    // collapse to one 10x10 map
        //       int rows = attn.Rows, cols = attn.Cols;
        //       for (int r = 0; r < rows; r++)
        //           for (int c = 0; c < cols; c++) {
        //               float v = map[r * cols + c]; // 0..1 attention weight
        //               // draw a cell colored/scaled by v ...
        //           }
        //   }
        //
        // --- Example C: a small derived vector ("activation_energy") ---
        //   EnactTensor energy = data.Tensor("encoder_out_norm"); // [10]
        //   if (energy != null) {
        //       for (int i = 0; i < energy.Length; i++) {
        //           float e = energy.Get(i);
        //           // ...
        //       }
        //   }
        //
        // Tip: data.Hz tells you the stream rate; frame.stamp is the timestamp.
        // =====================================================================
    }
}
