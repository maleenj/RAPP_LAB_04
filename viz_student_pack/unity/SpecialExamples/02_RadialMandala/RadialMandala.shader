// RadialMandala.shader (Enact/RadialMandala) — crisp single-colour emissive with a
// fresnel edge-glow rim. Built-in Render Pipeline. Opaque + ZWrite so the staggered
// rings occlude correctly and read with real depth/parallax. Push _Intensity above
// 1 and add Bloom for the glow. _Flash is spiked by the script on threshold crosses.
//
// (URP/HDRP: this is a Built-in-RP shader and will show magenta there — use a
// Built-in RP project for this visual, same as Special Example 01.)

Shader "Enact/RadialMandala"
{
    Properties
    {
        [HDR] _Color ("Color", Color) = (0.1, 0.7, 1.0, 1.0)
        _Intensity ("Intensity", Float) = 1.5
        _EdgeGlow ("Edge Glow", Range(0,8)) = 2.5
        _Flash ("Flash", Range(0,16)) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Cull Off
        ZWrite On

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata { float4 vertex : POSITION; float3 normal : NORMAL; };
            struct v2f
            {
                float4 pos       : SV_POSITION;
                float3 worldNorm : TEXCOORD0;
                float3 viewDir   : TEXCOORD1;
            };

            float4 _Color;
            float  _Intensity;
            float  _EdgeGlow;
            float  _Flash;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                float3 wp = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldNorm = UnityObjectToWorldNormal(v.normal);
                o.viewDir = normalize(_WorldSpaceCameraPos - wp);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 n = normalize(i.worldNorm);
                // Cull Off: faces pointing away flip the normal — use abs so the rim
                // term is stable on both sides of the thin tube.
                float ndv = saturate(abs(dot(n, normalize(i.viewDir))));
                float fres = pow(1.0 - ndv, 3.0);            // bright silhouette rim

                float3 col = _Color.rgb * _Intensity;        // base body
                col += _Color.rgb * fres * _EdgeGlow;        // edge glow
                col += _Color.rgb * _Flash;                  // accent flash spike
                return fixed4(col, 1.0);
            }
            ENDCG
        }
    }
    Fallback Off
}
