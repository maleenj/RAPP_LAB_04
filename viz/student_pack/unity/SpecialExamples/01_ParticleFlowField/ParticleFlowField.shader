// ParticleFlowField.shader — additive, camera-facing point sprites drawn straight
// from the GPU particle buffer (no mesh). Colour comes from a gradient ramp,
// indexed by particle speed; alpha fades in/out with particle life.
//
// Built-in Render Pipeline. (URP/HDRP users: see the README note.)

Shader "Enact/ParticleFlowField"
{
    Properties
    {
        _ColorRamp ("Colour Ramp", 2D) = "white" {}
        _Size      ("Size", Float) = 0.03
        _Intensity ("Intensity (glow)", Float) = 2
        _SpeedMax  ("Speed for max colour", Float) = 3
        _LifeMax   ("Life max", Float) = 4
        _Softness  ("Edge softness", Float) = 2
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Blend SrcAlpha One
        ZWrite Off
        Cull Off
        ZTest LEqual

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            #include "UnityCG.cginc"

            struct Particle { float3 pos; float3 vel; float life; float seed; };
            StructuredBuffer<Particle> _Particles;

            sampler2D _ColorRamp;
            float _Size, _Intensity, _SpeedMax, _LifeMax, _Softness;

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 uv  : TEXCOORD0;
                float4 col : TEXCOORD1;
            };

            static const float2 QUAD[6] = {
                float2(-1,-1), float2(1,-1), float2(1,1),
                float2(-1,-1), float2(1,1), float2(-1,1)
            };

            v2f vert(uint vid : SV_VertexID)
            {
                uint pid = vid / 6u;
                uint c   = vid % 6u;
                Particle p = _Particles[pid];

                float2 q     = QUAD[c];
                float3 right = UNITY_MATRIX_V[0].xyz;     // camera right (world)
                float3 up    = UNITY_MATRIX_V[1].xyz;     // camera up (world)
                float3 wpos  = p.pos + (right * q.x + up * q.y) * _Size;

                v2f o;
                o.pos = mul(UNITY_MATRIX_VP, float4(wpos, 1.0));
                o.uv  = q;

                float speed = length(p.vel);
                float t = saturate(speed / max(_SpeedMax, 1e-3));
                float4 ramp = tex2Dlod(_ColorRamp, float4(t, 0.5, 0, 0));

                float lifeFrac = saturate(p.life / max(_LifeMax, 1e-3));
                float fade = smoothstep(0.0, 0.15, lifeFrac);   // soft birth/death
                o.col = float4(ramp.rgb * _Intensity, ramp.a * fade);
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float d = length(i.uv);
                float a = saturate(1.0 - d);
                a = pow(a, _Softness);
                return fixed4(i.col.rgb, i.col.a * a);
            }
            ENDCG
        }
    }
}
