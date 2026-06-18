// ConnectionStatusUI.cs — on-screen connection status, rate, and latest values.
//
// Zero setup: attach to any GameObject and assign `client`. It draws a small
// overlay via OnGUI (no Canvas needed) so you can confirm at a glance that data
// is flowing. Handy on workshop day 1 while wiring up laptops.

using System.Text;
using UnityEngine;

public class ConnectionStatusUI : MonoBehaviour
{
    public VamClient client;

    [Tooltip("Channel whose latest values to show.")]
    public string channel = "robot_joint_states";

    void Start()
    {
        if (client == null) client = FindObjectOfType<VamClient>();
    }

    void OnGUI()
    {
        if (client == null) return;

        var sb = new StringBuilder();
        bool ok = client.IsConnected;
        sb.AppendLine(ok ? "● CONNECTED" : "○ disconnected");
        sb.AppendLine($"rate: {client.Hz:F1} Hz");
        sb.AppendLine($"channels: {client.LatestByChannel.Count}");

        if (client.LatestByChannel.TryGetValue(channel, out var frame) && frame.data != null)
        {
            sb.Append(channel).Append(": ");
            int n = Mathf.Min(frame.data.Length, 6);
            for (int i = 0; i < n; i++) sb.Append(frame.data[i].ToString("F2")).Append(' ');
            sb.AppendLine();
        }

        var style = new GUIStyle(GUI.skin.box)
        {
            alignment = TextAnchor.UpperLeft,
            fontSize = 16,
            normal = { textColor = ok ? Color.green : Color.red }
        };
        GUI.Box(new Rect(10, 10, 360, 120), sb.ToString(), style);
    }
}
