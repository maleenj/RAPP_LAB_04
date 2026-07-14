// EnactClient.cs — the shared connection to the ENACT data stream.
//
// You usually add ONE EnactClient to your scene. Everything else (EnactData,
// EnactInspector, your visuals) finds it automatically via EnactClient.Instance and
// shares this single connection. It parses EVERY channel — flat vectors AND
// nested tensors (attention matrices) — into a uniform EnactFrame.
//
// Two data sources, ONE code path:
//   * Live / replay : connect to ws://<host-ip>:8765 (the live bridge or player.py)
//   * Offline file  : play back a recorded .jsonl TextAsset, no server/network
//
// Live mode needs the free NativeWebSocket package:
//   Window > Package Manager > + > Add package from git URL:
//   https://github.com/endel/NativeWebSocket.git#upm
// File-playback works WITHOUT it (comment out USE_NATIVE_WEBSOCKET below).
//
// No JSON package needed — EnactJson.cs (bundled) handles parsing.

#define USE_NATIVE_WEBSOCKET   // comment out if you only do offline file playback

using System;
using System.Collections.Generic;
using UnityEngine;
#if USE_NATIVE_WEBSOCKET
using NativeWebSocket;
#endif

public class EnactClient : MonoBehaviour
{
    /// Shared instance — other scripts use EnactClient.Instance, no wiring needed.
    public static EnactClient Instance { get; private set; }

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

    /// Fired for every received frame (main thread).
    public event Action<EnactFrame> OnFrame;

    /// Latest frame per channel, for pollers.
    public readonly Dictionary<string, EnactFrame> LatestByChannel = new Dictionary<string, EnactFrame>();

    int _frameCount;
    float _rateTimer;

#if USE_NATIVE_WEBSOCKET
    WebSocket _ws;
#endif

    void Awake()
    {
        if (Instance == null) Instance = this;
        else if (Instance != this) Debug.LogWarning("[EnactClient] more than one EnactClient in the scene.");
    }

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
        _rateTimer += Time.deltaTime;
        if (_rateTimer >= 1f)
        {
            Hz = _frameCount / _rateTimer;
            _frameCount = 0;
            _rateTimer = 0f;
        }
    }

    // ---- parsing (handles flat AND tensor channels) ----------------------- //
    void Dispatch(string json)
    {
        EnactJson root = EnactJson.Parse(json);
        if (root.IsNull) return;
        string channel = root["channel"].AsString;
        if (string.IsNullOrEmpty(channel)) return;
        if (channel == "__status__") { IsConnected = true; return; }

        var frame = new EnactFrame { channel = channel, stamp = root["stamp"].AsDouble };

        if (root.Has("tensors"))
        {
            var node = root["tensors"];
            frame.tensors = new Dictionary<string, EnactTensor>();
            foreach (var name in node.Keys)
            {
                var t = node[name];
                frame.tensors[name] = new EnactTensor
                {
                    shape = t["shape"].AsIntArray(),
                    data = t["data"].AsFloatArray(),
                };
            }
        }
        if (root.Has("data"))
        {
            frame.data = root["data"].AsFloatArray();
            frame.shape = root["shape"].AsIntArray();
            frame.labels = root["labels"].AsStringArray();
        }

        IsConnected = true;
        _frameCount++;
        LatestByChannel[channel] = frame;
        OnFrame?.Invoke(frame);
    }

    // ---- WebSocket mode --------------------------------------------------- //
#if USE_NATIVE_WEBSOCKET
    async System.Threading.Tasks.Task ConnectWebSocket()
    {
        _ws = new WebSocket(url);
        _ws.OnOpen += () => { IsConnected = true; Debug.Log($"[EnactClient] connected: {url}"); };
        _ws.OnError += (e) => Debug.LogWarning($"[EnactClient] error: {e}");
        _ws.OnClose += (c) => { IsConnected = false; Debug.Log("[EnactClient] closed"); };
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
            Debug.LogError("[EnactClient] FilePlayback selected but no recording assigned.");
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

        Debug.Log("[EnactClient] file playback finished");
    }

    // Lightweight numeric extraction for replay pacing.
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
