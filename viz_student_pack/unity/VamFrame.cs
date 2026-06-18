// VamFrame.cs — the uniform data model every channel arrives as.
//
// You rarely construct these yourself; VamClient builds them from the network and
// hands them to you via VamData. You just READ the fields/helpers below.
//
// Two shapes of channel:
//   * Flat channels (joint_states, joint_targets, robot_joint_states):
//       use `data` (float[]), `shape` (int[]), `labels` (string[]).
//   * Tensor channels (activations, attn_entropy, activation_energy):
//       use `tensors["name"]` -> VamTensor (e.g. "decoder_xattn", "encoder_out").

using System.Collections.Generic;

/// One named numeric tensor (a sub-array of a tensor channel).
public class VamTensor
{
    public int[] shape;     // e.g. [2,4,10,10] or [10,128] or [10]
    public float[] data;    // flat, row-major

    public int Length => data != null ? data.Length : 0;

    /// Rows / Cols use the LAST two dimensions (the matrix you usually want to draw).
    public int Cols => (shape != null && shape.Length >= 1) ? shape[shape.Length - 1] : 0;
    public int Rows => (shape != null && shape.Length >= 2) ? shape[shape.Length - 2] : 1;
    /// How many [Rows x Cols] planes are stacked (e.g. layers*heads).
    public int Planes => (Rows * Cols) > 0 ? Length / (Rows * Cols) : 0;

    public float Get(int i) => (data != null && i >= 0 && i < data.Length) ? data[i] : 0f;

    /// Index into the last plane's [row, col].
    public float Get(int row, int col) => Get(row * Cols + col);

    /// Index into a specific plane's [row, col] (plane = layer/head index).
    public float Get(int plane, int row, int col) => Get(plane * Rows * Cols + row * Cols + col);

    /// Average all leading planes into a single [Rows x Cols] matrix (handy for
    /// collapsing [layers,heads,L,S] attention into one heatmap).
    public float[] MeanPlane()
    {
        int rc = Rows * Cols;
        var outp = new float[rc];
        if (Planes == 0) return outp;
        for (int p = 0; p < Planes; p++)
            for (int k = 0; k < rc; k++)
                outp[k] += data[p * rc + k] / Planes;
        return outp;
    }
}

/// One message on one channel.
public class VamFrame
{
    public string channel;
    public double stamp;

    // Flat-channel fields:
    public float[] data;      // the numbers
    public int[] shape;       // their shape (e.g. [6] or [3,4,10,10])
    public string[] labels;   // optional per-element names (e.g. joint names)

    // Tensor-channel field:
    public Dictionary<string, VamTensor> tensors;  // null for flat channels

    public bool HasTensors => tensors != null && tensors.Count > 0;
    public bool IsMatrix => shape != null && shape.Length >= 2;
    public int Length => data != null ? data.Length : 0;

    public int Cols => (shape != null && shape.Length >= 1) ? shape[shape.Length - 1] : 0;
    public int Rows => (shape != null && shape.Length >= 2) ? shape[shape.Length - 2] : 1;

    public float Get(int i) => (data != null && i >= 0 && i < data.Length) ? data[i] : 0f;
    public float Get(int row, int col) => Get(row * Cols + col);

    /// Convenience: a named tensor, or null if this channel/name has none.
    public VamTensor Tensor(string name) =>
        (tensors != null && tensors.TryGetValue(name, out var t)) ? t : null;
}
