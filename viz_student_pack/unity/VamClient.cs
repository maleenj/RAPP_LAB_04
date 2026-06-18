// VamClient.cs — receive the VAM data stream in Unity.
//
// Two data sources, ONE code path:
//   * Live / replay  : connect to ws://<host-ip>:8765 (the live bridge or player.py)
//   * Offline file   : play back a recorded .jsonl TextAsset, no server/network
//
// Other scripts subscribe to OnFrame to get every frame, or poll LatestByChannel.
//
// Requires the free NativeWebSocket package for the WebSocket mode:
//   Window > Package Manager > + > Add package from git URL:
//   https://github.com/endel/NativeWebSocket.git#upm
// (File-playback mode works WITHOUT NativeWebSocket — see USE_NATIVE_WEBSOCKET.)

#define USE_NATIVE_WEBSOCKET   // comment this line out if you only do file playback

using System;
using System.Collections.Generic;
using UnityEngine;
#if USE_NATIVE_WEBSOCKET
using NativeWebSocket;
#endif

public class VamClient : MonoBehaviour
{
    public enum Source { WebSocket, FilePlayback }

    [Header("Where the data comes from")]
    public Source source = Source.WebSocket;

    [Header("WebSocket mode")]
    [Tooltip("Live bridge or player.py. Use the host IP the instructor gives you.")]
    public string url = "ws://localhost:8765";

    [Header("File-playback mode (offline, no network)")]
    [Tooltip("A recorded .jsonl placed in the project (e.g. Assets/recordings/r1g1.jsonl.txt).")]
    public TextAsset recording;
    public bool loop = true;
    public float speed = 1.0f;

    [Header("Status (read-only)")]
    public bool IsConnected;
    public float Hz;

    // One uniform frame shape for every channel. Extra keys (t_recv, velocity,
    // tensors) are simply ignored by JsonUtility — the joints test only needs these.
    [Serializable]
    public class Frame
    {
        public string channel;
        public double stamp;
        public int[] shape;
        public float[] data;
        public string[] labels;
    }

    /// Fired for every received frame (main thread).
    public event Action<Frame> OnFrame;

    /// Latest frame per channel, for pollers.
    public readonly Dictionary<string, Frame> LatestByChannel = new Dictionary<string, Frame>();

    int _frameCount;
    float _rateTimer;

#if USE_NATIVE_WEBSOCKET
    WebSocket _ws;
#endif

    async void Start()
    {
        if (source == Source.WebSocket)
        {
#if USE_NATIVE_WEBSOCKET
            await ConnectWebSocket();
#else
            Debug.LogError("WebSocket mode needs NativeWebSocket. Enable USE_NATIVE_WEBSOCKET " +
                           "and install the package, or switch source to FilePlayback.");
#endif
        }
        else
        {
            StartCoroutine(PlayFromFile());
        }
    }

    void Update()
    {
#if USE_NATIVE_WEBSOCKET
        if (_ws != null)
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
            _ws.DispatchMessageQueue();
            #endif
        }
#endif
        // frame-rate estimate
        _rateTimer += Time.deltaTime;
        if (_rateTimer >= 1f)
        {
            Hz = _frameCount / _rateTimer;
            _frameCount = 0;
            _rateTimer = 0f;
        }
    }

    void Dispatch(string json)
    {
        Frame frame;
        try { frame = JsonUtility.FromJson<Frame>(json); }
        catch { return; }
        if (frame == null || string.IsNullOrEmpty(frame.channel)) return;
        if (frame.channel == "__status__") { IsConnected = true; return; }

        IsConnected = true;
        _frameCount++;
        LatestByChannel[frame.channel] = frame;
        OnFrame?.Invoke(frame);
    }

    // ---- WebSocket mode --------------------------------------------------- //
#if USE_NATIVE_WEBSOCKET
    async System.Threading.Tasks.Task ConnectWebSocket()
    {
        _ws = new WebSocket(url);
        _ws.OnOpen += () => { IsConnected = true; Debug.Log($"[VamClient] connected: {url}"); };
        _ws.OnError += (e) => Debug.LogWarning($"[VamClient] error: {e}");
        _ws.OnClose += (c) => { IsConnected = false; Debug.Log("[VamClient] closed"); };
        _ws.OnMessage += (bytes) => Dispatch(System.Text.Encoding.UTF8.GetString(bytes));
        await _ws.Connect();
    }

    async void OnApplicationQuit()
    {
        if (_ws != null) await _ws.Close();
    }
#endif

    // ---- File-playback mode (offline) ------------------------------------- //
    System.Collections.IEnumerator PlayFromFile()
    {
        if (recording == null)
        {
            Debug.LogError("[VamClient] FilePlayback selected but no recording assigned.");
            yield break;
        }
        var lines = recording.text.Split('\n');
        IsConnected = true;

        do
        {
            double prev = double.NaN;
            foreach (var raw in lines)
            {
                var line = raw.Trim();
                if (line.Length == 0) continue;

                // pull t_recv for pacing (JsonUtility Frame doesn't include it)
                double t = ExtractTRecv(line);
                if (!double.IsNaN(prev) && !double.IsNaN(t))
                {
                    float delay = (float)(t - prev) / Mathf.Max(speed, 0.0001f);
                    if (delay > 0f) yield return new WaitForSeconds(Mathf.Min(delay, 5f));
                }
                else
                {
                    yield return new WaitForSeconds((1f / 15f) / Mathf.Max(speed, 0.0001f));
                }
                if (!double.IsNaN(t)) prev = t;

                Dispatch(line);
            }
            yield return new WaitForSeconds(0.2f);
        } while (loop);

        Debug.Log("[VamClient] file playback finished");
    }

    // Lightweight numeric extraction so we don't need a JSON lib just for one field.
    static double ExtractTRecv(string line)
    {
        const string key = "\"t_recv\":";
        int i = line.IndexOf(key, StringComparison.Ordinal);
        if (i < 0) return double.NaN;
        i += key.Length;
        int j = i;
        while (j < line.Length && (char.IsDigit(line[j]) || line[j] == '.' || line[j] == '-' ||
                                   line[j] == 'e' || line[j] == 'E' || line[j] == '+')) j++;
        return double.TryParse(line.Substring(i, j - i),
            System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var v) ? v : double.NaN;
    }
}
