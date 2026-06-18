// VamData.cs — your data input. THIS is the only VAM class most students need.
//
// HOW TO USE
//   1. Make sure ONE VamClient exists in the scene (the connection).
//   2. Add this VamData component to your GameObject and type a `channel` name
//      in the Inspector (see the list below).
//   3. In your own script, read the data and build your visual:
//
//        VamData data = GetComponent<VamData>();
//        if (data.HasData) {
//            float[] v = data.Values;            // the raw numbers
//            float a   = data.Get(2, 3);         // matrix element [row,col]
//            VamTensor attn = data.Tensor("decoder_xattn");  // activation matrices
//        }
//
//   ...or subscribe to the event instead of polling:
//        data.OnData += frame => { /* new frame arrived */ };
//
// AVAILABLE CHANNELS (pick one for `channel`)
//   Flat vectors:
//     "robot_joint_states" — real robot joints [6] (has labels)
//     "joint_states"       — VAM predicted "ghost" joints [6]
//     "joint_targets"      — normalized targets [6]
//   Tensor channels (use data.Tensor("...")):
//     "activations"        -> decoder_xattn [2,4,10,10], encoder_selfattn [3,4,10,10],
//                             encoder_out [10,128], decoder_out [10,128],
//                             input_saliency [10,K] (if saliency enabled)
//     "attn_entropy"       -> decoder_xattn_entropy [2], encoder_selfattn_entropy [3]
//     "activation_energy"  -> encoder_out_norm [10], decoder_out_norm [10]
// (joint values are in radians.)

using System;
using UnityEngine;

public class VamData : MonoBehaviour
{
    [Tooltip("Which data stream to read. See the list at the top of VamData.cs.")]
    public string channel = "robot_joint_states";

    /// True once at least one frame for this channel has arrived.
    public bool HasData => Frame != null;

    /// The most recent frame for this channel (null until data arrives).
    public VamFrame Frame { get; private set; }

    /// Shortcut to the raw numbers of a flat channel (null for tensor channels).
    public float[] Values => Frame != null ? Frame.data : null;

    /// Update rate of the shared connection (Hz).
    public float Hz => _client != null ? _client.Hz : 0f;

    /// Raised on the main thread each time a frame arrives for this channel.
    public event Action<VamFrame> OnData;

    VamClient _client;

    void OnEnable()
    {
        _client = VamClient.Instance != null ? VamClient.Instance : FindObjectOfType<VamClient>();
        if (_client == null)
        {
            Debug.LogError("[VamData] No VamClient in the scene. Add one (the connection).");
            return;
        }
        _client.OnFrame += HandleFrame;
        // Seed with whatever's already cached so we don't wait for the next frame.
        if (_client.LatestByChannel.TryGetValue(channel, out var f)) Frame = f;
    }

    void OnDisable()
    {
        if (_client != null) _client.OnFrame -= HandleFrame;
    }

    void HandleFrame(VamFrame frame)
    {
        if (frame.channel != channel) return;
        Frame = frame;
        OnData?.Invoke(frame);
    }

    // ---- convenience pass-throughs ---------------------------------------- //

    /// Element i of a flat channel.
    public float Get(int i) => Frame != null ? Frame.Get(i) : 0f;

    /// Element [row, col] of a flat matrix channel.
    public float Get(int row, int col) => Frame != null ? Frame.Get(row, col) : 0f;

    /// A named tensor from a tensor channel (e.g. "decoder_xattn"), or null.
    public VamTensor Tensor(string name) => Frame != null ? Frame.Tensor(name) : null;
}
