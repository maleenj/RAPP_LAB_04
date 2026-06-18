// VamJson.cs — tiny, self-contained JSON parser (no external packages).
//
// Why this exists: Unity's built-in JsonUtility can't parse objects with dynamic
// keys (our `tensors` field, e.g. attention matrices). This minimal parser does,
// so EVERY channel — flat vectors and nested matrices — is readable with zero
// extra dependencies. You normally don't touch this file; VamClient uses it for you.
//
// Supports: objects, arrays, strings, numbers, true/false/null. Good enough for
// the VAM frame format. Not a general-purpose validator.

using System.Collections.Generic;
using System.Globalization;
using System.Text;

public class VamJson
{
    // Underlying value is one of: Dictionary<string,VamJson>, List<VamJson>,
    // string, double, bool, or null.
    object _v;

    public static readonly VamJson Null = new VamJson { _v = null };

    public bool IsNull => _v == null;
    public bool IsObject => _v is Dictionary<string, VamJson>;
    public bool IsArray => _v is List<VamJson>;

    // ---- accessors --------------------------------------------------------- //
    public VamJson this[string key]
    {
        get
        {
            if (_v is Dictionary<string, VamJson> d && d.TryGetValue(key, out var n)) return n;
            return Null;
        }
    }

    public VamJson this[int i]
    {
        get
        {
            if (_v is List<VamJson> l && i >= 0 && i < l.Count) return l[i];
            return Null;
        }
    }

    public int Count
    {
        get
        {
            if (_v is List<VamJson> l) return l.Count;
            if (_v is Dictionary<string, VamJson> d) return d.Count;
            return 0;
        }
    }

    public bool Has(string key) => _v is Dictionary<string, VamJson> d && d.ContainsKey(key);

    public IEnumerable<string> Keys =>
        _v is Dictionary<string, VamJson> d ? d.Keys : new List<string>();

    public double AsDouble => _v is double n ? n : 0.0;
    public float AsFloat => (float)AsDouble;
    public int AsInt => (int)AsDouble;
    public bool AsBool => _v is bool b && b;
    public string AsString => _v as string ?? "";

    public float[] AsFloatArray()
    {
        if (!(_v is List<VamJson> l)) return new float[0];
        var arr = new float[l.Count];
        for (int i = 0; i < l.Count; i++) arr[i] = l[i].AsFloat;
        return arr;
    }

    public int[] AsIntArray()
    {
        if (!(_v is List<VamJson> l)) return new int[0];
        var arr = new int[l.Count];
        for (int i = 0; i < l.Count; i++) arr[i] = l[i].AsInt;
        return arr;
    }

    public string[] AsStringArray()
    {
        if (!(_v is List<VamJson> l)) return new string[0];
        var arr = new string[l.Count];
        for (int i = 0; i < l.Count; i++) arr[i] = l[i].AsString;
        return arr;
    }

    // ---- parsing ----------------------------------------------------------- //
    public static VamJson Parse(string s)
    {
        int i = 0;
        try
        {
            var node = ParseValue(s, ref i);
            return node ?? Null;
        }
        catch
        {
            return Null;
        }
    }

    static void SkipWs(string s, ref int i)
    {
        while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }

    static VamJson ParseValue(string s, ref int i)
    {
        SkipWs(s, ref i);
        if (i >= s.Length) return Null;
        char c = s[i];
        if (c == '{') return ParseObject(s, ref i);
        if (c == '[') return ParseArray(s, ref i);
        if (c == '"') return new VamJson { _v = ParseString(s, ref i) };
        if (c == 't' || c == 'f') return ParseBool(s, ref i);
        if (c == 'n') { i += 4; return Null; } // null
        return ParseNumber(s, ref i);
    }

    static VamJson ParseObject(string s, ref int i)
    {
        var d = new Dictionary<string, VamJson>();
        i++; // {
        SkipWs(s, ref i);
        if (i < s.Length && s[i] == '}') { i++; return new VamJson { _v = d }; }
        while (i < s.Length)
        {
            SkipWs(s, ref i);
            string key = ParseString(s, ref i);
            SkipWs(s, ref i);
            if (i < s.Length && s[i] == ':') i++;
            var val = ParseValue(s, ref i);
            d[key] = val;
            SkipWs(s, ref i);
            if (i < s.Length && s[i] == ',') { i++; continue; }
            if (i < s.Length && s[i] == '}') { i++; break; }
            break;
        }
        return new VamJson { _v = d };
    }

    static VamJson ParseArray(string s, ref int i)
    {
        var l = new List<VamJson>();
        i++; // [
        SkipWs(s, ref i);
        if (i < s.Length && s[i] == ']') { i++; return new VamJson { _v = l }; }
        while (i < s.Length)
        {
            var val = ParseValue(s, ref i);
            l.Add(val);
            SkipWs(s, ref i);
            if (i < s.Length && s[i] == ',') { i++; continue; }
            if (i < s.Length && s[i] == ']') { i++; break; }
            break;
        }
        return new VamJson { _v = l };
    }

    static string ParseString(string s, ref int i)
    {
        var sb = new StringBuilder();
        i++; // opening quote
        while (i < s.Length)
        {
            char c = s[i++];
            if (c == '"') break;
            if (c == '\\' && i < s.Length)
            {
                char e = s[i++];
                switch (e)
                {
                    case 'n': sb.Append('\n'); break;
                    case 't': sb.Append('\t'); break;
                    case 'r': sb.Append('\r'); break;
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'u':
                        if (i + 4 <= s.Length)
                        {
                            int code = int.Parse(s.Substring(i, 4), NumberStyles.HexNumber,
                                                 CultureInfo.InvariantCulture);
                            sb.Append((char)code);
                            i += 4;
                        }
                        break;
                    default: sb.Append(e); break;
                }
            }
            else sb.Append(c);
        }
        return sb.ToString();
    }

    static VamJson ParseNumber(string s, ref int i)
    {
        int start = i;
        while (i < s.Length)
        {
            char c = s[i];
            if (char.IsDigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') i++;
            else break;
        }
        double.TryParse(s.Substring(start, i - start), NumberStyles.Float,
                        CultureInfo.InvariantCulture, out double val);
        return new VamJson { _v = val };
    }

    static VamJson ParseBool(string s, ref int i)
    {
        if (s[i] == 't') { i += 4; return new VamJson { _v = true }; }   // true
        i += 5; return new VamJson { _v = false };                        // false
    }
}
