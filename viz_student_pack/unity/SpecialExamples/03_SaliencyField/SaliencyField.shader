// SaliencyField.shader (Enact/SaliencyField) — flat unlit emissive for the heat
// cells. Per-cell colour comes from a MaterialPropertyBlock (_Color); _Intensity is
// a global multiplier so you can push the field above 1.0 for Bloom. Cull Off so a
// tilted cell stays visible from both sides. Built-in Render Pipeline.
//
// (URP/HDRP: Built-in-RP shader — shows magenta there; use a Built-in RP project.)

Shader "Enact/SaliencyField"
{
    Properties
    {
        [HDR] _Color ("Color", Color) = (1,1,1,1)
        _Intensity ("Intensity", Float) = 1.4
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

            struct appdata { float4 vertex : POSITION; };
            struct v2f { float4 pos : SV_POSITION; };

            float4 _Color;      // overridden per-cell via MaterialPropertyBlock
            float  _Intensity;  // global multiplier (on the shared material)

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return fixed4(_Color.rgb * _Intensity, 1.0);
            }
            ENDCG
        }
    }
    Fallback Off
}
