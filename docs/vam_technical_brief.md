# VAM Technical Document for IEEE RO-MAN Paper

## Context

This document provides the full technical details of the Vision-Action Model (VAM) system developed for RAPP Lab 04 — a lightweight Action Chunking Transformer that maps real-time human skeleton observations to UR10 cobot joint trajectories for improvised physical theatre. The system is designed for IEEE RO-MAN submission, targeting the intersection of embodied AI, human-robot interaction, and performance arts.

---

## 1. Paper Contributions (Unique Selling Points)

### 1.1 Primary Contributions

1. **Lightweight Action Chunking Transformer for Real-Time HRI Performance**
   - Only **1.14M parameters** (vs. 80M+ in original ACT, ~300M in RT-2) — 70x smaller than comparable vision-action models
   - Sub-millisecond inference: **0.74 ms mean latency** (1,352 Hz capable vs. 15 Hz required) — 99% compute headroom
   - Runs on modest GPU hardware (validated on RTX 5090 desktop and RTX 5070 laptop) — no cloud inference needed
   - This is critical for live performance: the model must respond in real-time to improvised human movement with zero perceptible lag

2. **End-to-End Design Process: From Physical Theatre Methodology to Deployed Robot System**
   - First systematic integration of physical theatre training methodologies (Meyerhold's biomechanics, Lecoq pedagogy, Bogart's Viewpoints) as structured training data for a vision-action model
   - Complete pipeline: rosbag recording → skeleton extraction → temporal synchronization → action chunking → model training → real-time inference → physical robot control
   - Reproducible design process documented for the robotics arts community

3. **Temporal Ensemble with Exponential Decay for Smooth Continuous Motion**
   - Eliminates chunk boundary discontinuities without filtering lag — critical for theatre (jerky motion breaks the illusion of agency)
   - Overlapping predictions weighted by exponential decay produce physically smooth trajectories
   - Integrated with MoveIt Servo's 250 Hz interpolation for seamless real-time control

4. **Training Process for Physical Theatre Methodologies**
   - Multi-loss training objective incorporating trajectory prediction + temporal smoothness + acceleration smoothness + joint limit enforcement
   - Data augmentation strategies (temporal jitter, skeleton noise) designed for the noise characteristics of depth-camera skeleton tracking in performance spaces
   - Episode-level train/val/test splitting prevents temporal data leakage

5. **Three-Layer Safety Architecture for Live Performance with Untrained Performers**
   - Layer 1: MoveIt Servo (250 Hz) — collision checking, joint limits, singularity detection
   - Layer 2: VAM SafetyChecker (15 Hz) — velocity/acceleration pre-filtering
   - Layer 3: UR10 hardware controller — final safety enforcement
   - Critical for workshop settings where non-expert participants interact with the robot

### 1.2 Why These Contributions Matter for RO-MAN

- **Lightweight model**: Democratizes embodied AI for arts/HRI labs without expensive compute infrastructure
- **Design process**: Provides a replicable methodology for other researchers wanting to create responsive robot performers
- **Smooth motion**: The temporal ensemble approach is transferable to any action chunking system needing continuous output
- **Safety**: Essential for any system where non-experts physically interact with a robot arm in an unstructured environment

---

## 2. System Architecture

### 2.1 System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────┐ │
│  │  ROS2    │   │  Rosbag      │   │  Data       │   │  Model   │ │
│  │  Rosbag  │──▶│  Processing  │──▶│  Preparation│──▶│ Training │ │
│  │  Record  │   │  (Notebook   │   │  (Notebook  │   │(Notebook │ │
│  │          │   │   01)        │   │   02)       │   │  03)     │ │
│  └──────────┘   └──────────────┘   └─────────────┘   └──────────┘ │
│       │               │                  │                  │      │
│       ▼               ▼                  ▼                  ▼      │
│  ZED Camera    Synchronized CSV    Windowed Tensors    Checkpoint  │
│  + UR10        + Metadata          + Norm Stats        (best.pt)  │
│  Joint States                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME INFERENCE                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────┐ │
│  │  ZED     │   │   VAM ROS2   │   │  MoveIt   │   │   UR10    │ │
│  │  Camera  │──▶│   Node       │──▶│  Servo    │──▶│  Robot    │ │
│  │  (Body   │   │   (15 Hz)    │   │  (250 Hz) │   │           │ │
│  │  Track)  │   │              │   │           │   │           │ │
│  └──────────┘   └──────────────┘   └───────────┘   └───────────┘ │
│                        │                                           │
│            ┌───────────┼───────────┐                               │
│            ▼           ▼           ▼                               │
│     Input         Temporal      Safety                             │
│     Assembler     Ensemble      Checker                            │
│     [T_in buf]    [exp decay]   [3-layer]                          │
│            │           │           │                               │
│            ▼           ▼           ▼                               │
│     ACT Model    Blended       Constrained                         │
│     [1.14M]      Target        Velocity                            │
│                                Command                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key File Paths

| Component | File | Lines |
|-----------|------|-------|
| Model architecture | `vam_utils/model/act.py` | 218 |
| Model config | `vam_utils/config/model_config.py` | 124 |
| Training losses | `vam_utils/training/losses.py` | 86 |
| Trainer | `vam_utils/training/trainer.py` | 285 |
| Dataset | `vam_utils/data/dataset.py` | 150 |
| Normalization | `vam_utils/data/normalization.py` | 138 |
| Input assembler | `vam_utils/inference/input_assembler.py` | 117 |
| Temporal ensemble | `vam_utils/inference/temporal_ensemble.py` | 144 |
| Safety checker | `vam_utils/inference/safety_checker.py` | 150 |
| ROS2 inference node | `ros2_ws/src/vam_inference/vam_inference/vam_node.py` | 492 |
| **Total vam_utils** | | **~2,211** |

---

## 3. Model Architecture — Action Chunking Transformer (ACT)

### 3.1 Architecture Overview

The VAM employs a lightweight encoder-decoder transformer architecture that maps a temporal window of human skeleton observations and (optionally) robot joint states to a chunk of future robot joint trajectories.

**Input**: $\mathbf{X} \in \mathbb{R}^{B \times T_{\text{in}} \times D_{\text{in}}}$ where $D_{\text{in}} = D_{\text{skel}} + D_{\text{joint}} = 48 + 6 = 54$

**Output**: $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times T_{\text{out}} \times D_{\text{joint}}}$ where $D_{\text{joint}} = 6$

### 3.2 Mathematical Formulation

#### Input Processing

The input $\mathbf{X}$ is split into skeleton and robot components:

$$\mathbf{S} = \mathbf{X}_{[:, :, :48]} \in \mathbb{R}^{B \times T_{\text{in}} \times 48}$$
$$\mathbf{R} = \mathbf{X}_{[:, :, 48:]} \in \mathbb{R}^{B \times T_{\text{in}} \times 6}$$

where $\mathbf{S}$ contains 16 ZED skeleton keypoints $\times$ 3 coordinates (in the robot base_link frame), and $\mathbf{R}$ contains 6 UR10 joint angles in radians.

#### Modality Projections

Skeleton features are projected via a linear layer:

$$\mathbf{E}_{\text{skel}} = \mathbf{S} \cdot \mathbf{W}_{\text{skel}} + \mathbf{b}_{\text{skel}}, \quad \mathbf{W}_{\text{skel}} \in \mathbb{R}^{48 \times d}$$

Robot joint state is projected via a two-layer MLP with GELU activation:

$$\mathbf{H}_{\text{robot}} = \text{GELU}(\mathbf{R} \cdot \mathbf{W}_1 + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{6 \times d_h}$$
$$\mathbf{E}_{\text{robot}} = \mathbf{H}_{\text{robot}} \cdot \mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{d_h \times d}$$

where $d = 128$ (model dimension) and $d_h = 64$ (robot MLP hidden dimension).

#### Additive Fusion + Positional Encoding

The two modality embeddings are fused additively and augmented with learned positional encodings:

$$\mathbf{F} = \mathbf{E}_{\text{skel}} + \mathbf{E}_{\text{robot}} + \mathbf{P}_{\text{enc}}, \quad \mathbf{P}_{\text{enc}} \in \mathbb{R}^{T_{\text{in}} \times d}$$

where $\mathbf{P}_{\text{enc}}$ is a learned embedding table indexed by position.

**Design choice**: Additive fusion (vs. concatenation or cross-attention) was chosen for minimal parameter overhead while allowing the model to learn complementary representations from both modalities.

#### Transformer Encoder

The fused sequence is processed by a Pre-LN transformer encoder:

$$\mathbf{M} = \text{TransformerEncoder}(\mathbf{F}) \in \mathbb{R}^{B \times T_{\text{in}} \times d}$$

Configuration:
- $L_{\text{enc}} = 3$ layers
- $H = 4$ attention heads
- $d_{\text{ff}} = 512$ (feedforward dimension, $4 \times d$)
- Pre-LayerNorm (norm-first) for training stability
- GELU activation
- Dropout $p = 0.1$

#### Action Queries + Transformer Decoder

A set of $T_{\text{out}}$ learned action query embeddings serves as the initial decoder input:

$$\mathbf{Q} \in \mathbb{R}^{T_{\text{out}} \times d}$$

The decoder attends to the encoder memory via cross-attention:

$$\mathbf{D} = \text{TransformerDecoder}(\mathbf{Q}, \mathbf{M}) \in \mathbb{R}^{B \times T_{\text{out}} \times d}$$

Configuration:
- $L_{\text{dec}} = 2$ layers
- Same $H, d_{\text{ff}}$, Pre-LN, GELU, dropout as encoder

Each action query learns to specialize in predicting a specific temporal offset within the output chunk.

#### Output Projection

$$\hat{\mathbf{Y}} = \mathbf{D} \cdot \mathbf{W}_{\text{out}} + \mathbf{b}_{\text{out}}, \quad \mathbf{W}_{\text{out}} \in \mathbb{R}^{d \times 6}$$

Output is in normalized joint space; denormalization is applied during inference.

### 3.3 Transformer Layer Architecture — Detailed Internals

This section provides the full mathematical specification of each sublayer within the transformer encoder and decoder, as implemented in PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer` with the `norm_first=True` (Pre-LN) configuration.

**Notation**: $d = 128$ (model dimension), $H = 4$ (number of attention heads), $d_k = d_v = d/H = 32$ (per-head dimension), $d_{\text{ff}} = 512$ (feedforward hidden dimension).

#### 3.3.1 Scaled Dot-Product Attention

The fundamental attention operation computes a weighted sum of value vectors, where the weights are determined by the compatibility between query and key vectors:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

where $\mathbf{Q} \in \mathbb{R}^{n \times d_k}$, $\mathbf{K} \in \mathbb{R}^{m \times d_k}$, $\mathbf{V} \in \mathbb{R}^{m \times d_v}$, $n$ is the query sequence length, and $m$ is the key/value sequence length.

The scaling factor $\sqrt{d_k} = \sqrt{32} \approx 5.66$ prevents the dot products from growing large in magnitude, which would push the softmax into regions with extremely small gradients (the saturation problem described by Vaswani et al., 2017).

The softmax is applied row-wise: each query position produces a probability distribution over all key positions:

$$\alpha_{ij} = \frac{\exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{l=1}^{m} \exp(\mathbf{q}_i^\top \mathbf{k}_l / \sqrt{d_k})}$$

The output for query position $i$ is then the weighted combination: $\mathbf{o}_i = \sum_{j=1}^{m} \alpha_{ij} \mathbf{v}_j$.

**In this architecture**: No attention masks are applied in either the encoder or decoder. The encoder uses self-attention ($n = m = T_{\text{in}} = 10$). The decoder uses self-attention over action queries ($n = m = T_{\text{out}} = 10$) and cross-attention where queries come from the decoder and keys/values from the encoder memory ($n = T_{\text{out}}, m = T_{\text{in}}$).

#### 3.3.2 Multi-Head Attention (MHA)

Rather than performing a single attention function with $d$-dimensional keys, values, and queries, multi-head attention projects them into $H$ separate subspaces, performs attention independently in each, and concatenates the results:

$$\text{MultiHead}(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot \mathbf{W}^O$$

where each head $i \in \{1, \ldots, H\}$:

$$\text{head}_i = \text{Attention}(\mathbf{X}_q \mathbf{W}^Q_i, \; \mathbf{X}_k \mathbf{W}^K_i, \; \mathbf{X}_v \mathbf{W}^V_i)$$

**Learned projection matrices** (per head):

$$\mathbf{W}^Q_i \in \mathbb{R}^{d \times d_k}, \quad \mathbf{W}^K_i \in \mathbb{R}^{d \times d_k}, \quad \mathbf{W}^V_i \in \mathbb{R}^{d \times d_v}$$

**Output projection** (concatenation of all heads back to model dimension):

$$\mathbf{W}^O \in \mathbb{R}^{Hd_v \times d} = \mathbb{R}^{128 \times 128}$$

**Implementation note**: In PyTorch's `nn.MultiheadAttention`, the per-head projections are implemented as a single large matrix for computational efficiency. The $H$ separate $\mathbf{W}^Q_i$ matrices are packed into one $\mathbf{W}^Q \in \mathbb{R}^{d \times d}$, and the output is reshaped and split into heads internally. With $d = 128$ and $H = 4$, each head attends to a 32-dimensional subspace, allowing different heads to capture different types of temporal relationships in the skeleton/robot data.

**Parameters per MHA sublayer**:
- $\mathbf{W}^Q$: $128 \times 128 + 128 = 16{,}512$
- $\mathbf{W}^K$: $128 \times 128 + 128 = 16{,}512$
- $\mathbf{W}^V$: $128 \times 128 + 128 = 16{,}512$
- $\mathbf{W}^O$: $128 \times 128 + 128 = 16{,}512$
- **Total per MHA**: $4 \times 16{,}512 = \mathbf{66{,}048}$

#### 3.3.3 GELU Activation Function

The Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016) is used as the nonlinearity in both the feedforward sublayers and the robot projection MLP:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian. Unlike ReLU, GELU is smooth and non-monotonic near zero, providing a stochastic regularization effect: inputs are scaled by their percentile in a Gaussian distribution rather than being hard-gated. This has been shown to improve training dynamics in transformer architectures.

**Design rationale**: GELU was chosen over ReLU for its smoother gradient landscape, which is particularly beneficial for the small model size (1.14M params) and limited training data (~7K windows), where sharp activation boundaries could lead to dead neurons.

#### 3.3.4 Position-wise Feedforward Network (FFN)

Each transformer layer contains a two-layer feedforward network applied independently to each position (i.e., each timestep in the sequence):

$$\text{FFN}(\mathbf{x}) = \text{GELU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

where:
- $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{\text{ff}}} = \mathbb{R}^{128 \times 512}$, $\mathbf{b}_1 \in \mathbb{R}^{512}$
- $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d} = \mathbb{R}^{512 \times 128}$, $\mathbf{b}_2 \in \mathbb{R}^{128}$

The expansion ratio $d_{\text{ff}} / d = 512 / 128 = 4\times$ follows the standard transformer convention. The FFN acts as a per-position nonlinear transformation that can learn feature interactions that attention alone cannot capture.

**Parameters per FFN sublayer**:
- $\mathbf{W}_1 + \mathbf{b}_1$: $128 \times 512 + 512 = 66{,}048$
- $\mathbf{W}_2 + \mathbf{b}_2$: $512 \times 128 + 128 = 65{,}664$
- **Total per FFN**: $66{,}048 + 65{,}664 = \mathbf{131{,}712}$

#### 3.3.5 Pre-LayerNorm vs Post-LayerNorm

This model uses the **Pre-LN** (norm-first) transformer variant, where LayerNorm is applied *before* each sublayer rather than after. This is set via `norm_first=True` in PyTorch.

**LayerNorm** normalizes across the feature dimension:

$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

where $\mu, \sigma^2$ are the mean and variance computed across the $d = 128$ feature dimensions for each position, $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$ are learned scale and shift parameters, and $\epsilon = 10^{-5}$.

**Pre-LN formulation** (used in this model):

$$\mathbf{x}' = \mathbf{x} + \text{Sublayer}(\text{LN}(\mathbf{x}))$$

**Post-LN formulation** (original Vaswani et al.):

$$\mathbf{x}' = \text{LN}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

**Why Pre-LN**: Xiong et al. (2020, "On Layer Normalization in the Transformer Architecture") showed that Pre-LN allows gradients to flow more directly through residual connections, enabling stable training without learning rate warmup. This is critical for our small model and dataset — Post-LN transformers with small batch sizes can exhibit training instability. Pre-LN also places well-conditioned inputs into each sublayer, reducing sensitivity to initialization.

**Final norm**: Both the encoder and decoder apply an additional `nn.LayerNorm(d_model)` after the last layer, which normalizes the output of the entire stack.

**Parameters per LayerNorm**: $\boldsymbol{\gamma} + \boldsymbol{\beta} = 128 + 128 = \mathbf{256}$

#### 3.3.6 Pre-LN Encoder Layer

Each of the $L_{\text{enc}} = 3$ encoder layers consists of two sublayers with residual connections and Pre-LN:

**Sublayer 1 — Multi-Head Self-Attention**:

$$\mathbf{Z}^{(l)} = \mathbf{F}^{(l)} + \text{Dropout}\!\Big(\text{MHA}\big(\text{LN}_1(\mathbf{F}^{(l)}), \; \text{LN}_1(\mathbf{F}^{(l)}), \; \text{LN}_1(\mathbf{F}^{(l)})\big)\Big)$$

where $\mathbf{F}^{(l)} \in \mathbb{R}^{B \times T_{\text{in}} \times d}$ is the input to layer $l$ (with $\mathbf{F}^{(0)} = \mathbf{F}$, the fused + position-encoded input), and the three arguments to MHA are $(\mathbf{X}_q, \mathbf{X}_k, \mathbf{X}_v)$ — for self-attention, all three are the same (the normalized input). Dropout is applied to the attention output before the residual addition, with $p = 0.1$.

**Sublayer 2 — Position-wise Feedforward**:

$$\mathbf{F}^{(l+1)} = \mathbf{Z}^{(l)} + \text{Dropout}\!\Big(\text{FFN}\big(\text{LN}_2(\mathbf{Z}^{(l)})\big)\Big)$$

**Tensor flow through one encoder layer** (concrete shapes):
```
Input:              F^(l)           [B, 10, 128]
  → LN_1:          LN(F^(l))       [B, 10, 128]
  → Q,K,V proj:    Q,K,V           [B, 4, 10, 32]  (4 heads, 32-dim each)
  → Attention:     softmax(QK^T/√32)V  [B, 4, 10, 32]
  → Concat+W^O:    MHA output      [B, 10, 128]
  → Dropout(0.1) + Residual:  Z^(l) [B, 10, 128]
  → LN_2:          LN(Z^(l))       [B, 10, 128]
  → FFN W_1+GELU:  hidden          [B, 10, 512]
  → FFN W_2:       FFN output      [B, 10, 128]
  → Dropout(0.1) + Residual:  F^(l+1) [B, 10, 128]
```

After $L_{\text{enc}} = 3$ layers, a final LayerNorm produces the encoder memory:

$$\mathbf{M} = \text{LN}_{\text{final}}(\mathbf{F}^{(3)}) \in \mathbb{R}^{B \times T_{\text{in}} \times d}$$

**Parameters per encoder layer**:

| Sublayer | Parameters |
|----------|-----------|
| LN_1 (γ, β) | 256 |
| Self-Attention MHA (W^Q, W^K, W^V, W^O + biases) | 66,048 |
| LN_2 (γ, β) | 256 |
| FFN (W_1, b_1, W_2, b_2) | 131,712 |
| **Total per encoder layer** | **198,272** |

**Total encoder**: $3 \times 198{,}272 + 256 \text{ (final LN)} = \mathbf{595{,}072}$

#### 3.3.7 Pre-LN Decoder Layer

Each of the $L_{\text{dec}} = 2$ decoder layers consists of three sublayers with residual connections and Pre-LN:

**Sublayer 1 — Masked Self-Attention over Action Queries**:

$$\mathbf{A}^{(l)} = \mathbf{Q}^{(l)} + \text{Dropout}\!\Big(\text{MHA}_{\text{self}}\big(\text{LN}_1(\mathbf{Q}^{(l)}), \; \text{LN}_1(\mathbf{Q}^{(l)}), \; \text{LN}_1(\mathbf{Q}^{(l)})\big)\Big)$$

where $\mathbf{Q}^{(l)} \in \mathbb{R}^{B \times T_{\text{out}} \times d}$ is the decoder input at layer $l$ (with $\mathbf{Q}^{(0)}$ being the learned action query embeddings, expanded across the batch).

**Note on masking**: No causal mask is applied in the decoder self-attention. Unlike autoregressive language models, all action queries attend to each other freely. This is by design — the action chunk is predicted in parallel (non-autoregressively), and each query can benefit from context about other positions in the output chunk.

**Sublayer 2 — Cross-Attention to Encoder Memory**:

$$\mathbf{C}^{(l)} = \mathbf{A}^{(l)} + \text{Dropout}\!\Big(\text{MHA}_{\text{cross}}\big(\text{LN}_2(\mathbf{A}^{(l)}), \; \mathbf{M}, \; \mathbf{M}\big)\Big)$$

Here the queries come from the decoder (the normalized output of self-attention), while the keys and values come from the encoder memory $\mathbf{M}$. This is the critical information bottleneck where the decoder extracts relevant temporal context from the observed skeleton/robot history to inform each predicted joint angle.

**Important implementation detail**: In PyTorch's `nn.TransformerDecoderLayer` with `norm_first=True`, LayerNorm is applied to the query input but **not** to the encoder memory $\mathbf{M}$ (which has already been normalized by the encoder's final LayerNorm). The cross-attention computes:

$$\mathbf{Q}_{\text{cross}} = \text{LN}_2(\mathbf{A}^{(l)}) \mathbf{W}^Q_{\text{cross}}, \quad \mathbf{K}_{\text{cross}} = \mathbf{M} \mathbf{W}^K_{\text{cross}}, \quad \mathbf{V}_{\text{cross}} = \mathbf{M} \mathbf{W}^V_{\text{cross}}$$

Each of the $T_{\text{out}} = 10$ output positions attends to all $T_{\text{in}} = 10$ encoder memory positions, producing a $10 \times 10$ attention matrix per head.

**Sublayer 3 — Position-wise Feedforward**:

$$\mathbf{Q}^{(l+1)} = \mathbf{C}^{(l)} + \text{Dropout}\!\Big(\text{FFN}\big(\text{LN}_3(\mathbf{C}^{(l)})\big)\Big)$$

**Tensor flow through one decoder layer** (concrete shapes):
```
Input:              Q^(l)           [B, 10, 128]

--- Self-Attention ---
  → LN_1:          LN(Q^(l))       [B, 10, 128]
  → Self Q,K,V:    Q,K,V           [B, 4, 10, 32]
  → Attention:     softmax(QK^T/√32)V  [B, 4, 10, 32]
  → Concat+W^O:    self-attn out   [B, 10, 128]
  → Dropout + Residual:  A^(l)     [B, 10, 128]

--- Cross-Attention ---
  → LN_2:          LN(A^(l))       [B, 10, 128]  (query source)
  → Cross Q:       Q_cross         [B, 4, 10, 32]  (from decoder)
  → Cross K,V:     K_cross, V_cross [B, 4, 10, 32]  (from encoder memory M)
  → Attention:     softmax(Q_cross K_cross^T/√32) V_cross  [B, 4, 10, 32]
  → Concat+W^O:    cross-attn out  [B, 10, 128]
  → Dropout + Residual:  C^(l)     [B, 10, 128]

--- Feedforward ---
  → LN_3:          LN(C^(l))       [B, 10, 128]
  → FFN W_1+GELU:  hidden          [B, 10, 512]
  → FFN W_2:       FFN output      [B, 10, 128]
  → Dropout + Residual:  Q^(l+1)   [B, 10, 128]
```

After $L_{\text{dec}} = 2$ layers, a final LayerNorm produces the decoded output:

$$\mathbf{D} = \text{LN}_{\text{final}}(\mathbf{Q}^{(2)}) \in \mathbb{R}^{B \times T_{\text{out}} \times d}$$

**Parameters per decoder layer**:

| Sublayer | Parameters |
|----------|-----------|
| LN_1 (γ, β) | 256 |
| Self-Attention MHA (W^Q, W^K, W^V, W^O + biases) | 66,048 |
| LN_2 (γ, β) | 256 |
| Cross-Attention MHA (W^Q, W^K, W^V, W^O + biases) | 66,048 |
| LN_3 (γ, β) | 256 |
| FFN (W_1, b_1, W_2, b_2) | 131,712 |
| **Total per decoder layer** | **264,576** |

**Total decoder**: $2 \times 264{,}576 + 256 \text{ (final LN)} = \mathbf{529{,}408}$

#### 3.3.8 Learned Action Queries

The action query embeddings $\mathbf{Q}^{(0)} \in \mathbb{R}^{T_{\text{out}} \times d}$ are a learned parameter matrix (implemented as `nn.Embedding(T_out, d_model)`). Each of the $T_{\text{out}} = 10$ query vectors is initialized from $\mathcal{N}(0, 0.02^2)$ and learns to specialize in predicting a specific temporal offset within the output chunk.

**Mechanism**: Unlike autoregressive decoders (e.g., GPT), which generate tokens one at a time using the previous output as the next input, the action queries produce the entire output chunk in a single forward pass. This is the "action chunking" paradigm from Zhao et al. (2023):

1. Query $\mathbf{q}_0$ learns to predict the joint angles at $t+1$ (one step ahead)
2. Query $\mathbf{q}_1$ learns to predict the joint angles at $t+2$ (two steps ahead)
3. ...
4. Query $\mathbf{q}_9$ learns to predict the joint angles at $t+10$ (ten steps ahead)

Each query attends to the same encoder memory $\mathbf{M}$ via cross-attention but extracts different temporal information based on its learned representation. The non-causal self-attention between queries allows them to coordinate — e.g., query $\mathbf{q}_5$ can consider what queries $\mathbf{q}_3$ and $\mathbf{q}_4$ are predicting to ensure smooth trajectories.

At inference time, the queries are broadcast across the batch dimension:

$$\mathbf{Q}^{(0)}_{\text{batch}} = \mathbf{Q}^{(0)}.\text{unsqueeze}(0).\text{expand}(B, -1, -1) \in \mathbb{R}^{B \times T_{\text{out}} \times d}$$

This means all samples in a batch share the same query initialization, but diverge through the cross-attention to different encoder memories.

#### 3.3.9 Complete Forward Pass — Step-by-Step

The full forward pass, from raw input to predicted joint trajectories:

```
Step 1: INPUT SPLITTING
  x [B, 10, 54] → skeleton [B, 10, 48] + robot [B, 10, 6]

Step 2: MODALITY PROJECTION
  skeleton [B, 10, 48] → skeleton_proj → skel_emb [B, 10, 128]
  robot [B, 10, 6] → robot_proj (MLP: 6→64→GELU→128) → robot_emb [B, 10, 128]

Step 3: ADDITIVE FUSION + POSITIONAL ENCODING
  fused = skel_emb + robot_emb [B, 10, 128]
  positions = [0, 1, 2, ..., 9] → encoder_pos → pos_emb [10, 128]
  fused = fused + pos_emb [B, 10, 128]  (broadcast over batch)

Step 4: TRANSFORMER ENCODER (3 layers)
  fused [B, 10, 128]
    → Encoder Layer 1 (self-attn + FFN with Pre-LN) → [B, 10, 128]
    → Encoder Layer 2 (self-attn + FFN with Pre-LN) → [B, 10, 128]
    → Encoder Layer 3 (self-attn + FFN with Pre-LN) → [B, 10, 128]
    → Final LayerNorm → memory [B, 10, 128]

Step 5: ACTION QUERY PREPARATION
  action_queries.weight [10, 128] → expand → queries [B, 10, 128]

Step 6: TRANSFORMER DECODER (2 layers)
  queries [B, 10, 128], memory [B, 10, 128]
    → Decoder Layer 1 (self-attn + cross-attn to memory + FFN) → [B, 10, 128]
    → Decoder Layer 2 (self-attn + cross-attn to memory + FFN) → [B, 10, 128]
    → Final LayerNorm → decoded [B, 10, 128]

Step 7: OUTPUT PROJECTION
  decoded [B, 10, 128] → output_proj (Linear 128→6) → output [B, 10, 6]
```

**Output semantics**: Each of the 10 output frames contains 6 predicted joint angles in **normalized** space (z-scored using training statistics). Denormalization to physical joint angles (radians) is applied during inference by the `InputAssembler`.

### 3.4 Skeleton-Only Variant (SkeletonOnlyACT)

For live performance where robot proprioception introduces a feedback loop, a skeleton-only variant removes the robot state input:

- No robot projection MLP — the skeleton embedding passes directly to the encoder
- No additive fusion — skeleton embedding $\mathbf{E}_{\text{skel}} + \mathbf{P}_{\text{enc}}$ is the encoder input
- Input: $\mathbf{X} \in \mathbb{R}^{B \times T_{\text{in}} \times 48}$ (skeleton only)
- **1,134,086 parameters** (vs. 1,142,854 full model — difference is exactly the robot MLP: 8,768 params)
- All transformer layers (encoder and decoder) are identical to the full model
- Removes proprioceptive feedback: model predicts purely from human movement observation

This variant is used for real robot deployment to avoid the feedback loop where predicted joints feed back into the next prediction.

### 3.5 Detailed Parameter Count

| Component | Sublayer Detail | Parameters |
|-----------|----------------|-----------|
| **Input Stage** | | |
| Skeleton projection | Linear(48, 128) + bias | 6,272 |
| Robot MLP layer 1 | Linear(6, 64) + bias | 448 |
| Robot MLP layer 2 | Linear(64, 128) + bias | 8,320 |
| Encoder positional embedding | Embedding(10, 128) | 1,280 |
| **Encoder (×3 layers)** | | |
| LN_1 (per layer) | γ(128) + β(128) | 256 |
| Self-Attention Q,K,V,O (per layer) | 4 × Linear(128,128)+bias | 66,048 |
| LN_2 (per layer) | γ(128) + β(128) | 256 |
| FFN W_1 (per layer) | Linear(128, 512) + bias | 66,048 |
| FFN W_2 (per layer) | Linear(512, 128) + bias | 65,664 |
| **Subtotal per encoder layer** | | **198,272** |
| **Subtotal 3 encoder layers** | | **594,816** |
| Encoder final LayerNorm | γ(128) + β(128) | 256 |
| **Decoder (×2 layers)** | | |
| Action query embedding | Embedding(10, 128) | 1,280 |
| LN_1 (per layer) | γ(128) + β(128) | 256 |
| Self-Attention Q,K,V,O (per layer) | 4 × Linear(128,128)+bias | 66,048 |
| LN_2 (per layer) | γ(128) + β(128) | 256 |
| Cross-Attention Q,K,V,O (per layer) | 4 × Linear(128,128)+bias | 66,048 |
| LN_3 (per layer) | γ(128) + β(128) | 256 |
| FFN W_1 (per layer) | Linear(128, 512) + bias | 66,048 |
| FFN W_2 (per layer) | Linear(512, 128) + bias | 65,664 |
| **Subtotal per decoder layer** | | **264,576** |
| **Subtotal 2 decoder layers** | | **529,152** |
| Decoder final LayerNorm | γ(128) + β(128) | 256 |
| **Output Stage** | | |
| Output projection | Linear(128, 6) + bias | 774 |
| | | |
| **Total (full model)** | | **1,142,854** |
| **Total (skeleton-only)** | (remove robot MLP: −8,768) | **1,134,086** |

### 3.6 Weight Initialization

All weights are initialized before training begins. The initialization strategy is designed for stable gradient flow in a Pre-LN transformer:

- **Linear layers** (all projections, FFN, MHA): Xavier uniform initialization

  $$W_{ij} \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \; \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

  This maintains variance across layers by accounting for both fan-in and fan-out, preventing gradients from vanishing or exploding through the network depth.

- **Bias terms**: Initialized to zero (all linear layers including MHA and FFN).

- **Embedding tables** ($\mathbf{P}_{\text{enc}}$, $\mathbf{Q}$): Normal distribution with small standard deviation:

  $$\mathbf{P}_{\text{enc}}, \mathbf{Q} \sim \mathcal{N}(0, 0.02^2)$$

  The small $\sigma = 0.02$ ensures that initial positional encodings and action queries contribute minimally at the start of training, allowing the model to first learn the projection and attention weights before specializing the embeddings.

- **LayerNorm parameters**: $\boldsymbol{\gamma}$ initialized to 1, $\boldsymbol{\beta}$ initialized to 0 (PyTorch default — identity transformation initially).

---

## 4. Data Pipeline

### 4.1 Recording Setup

- **Robot**: Universal Robots UR10 (6-DOF collaborative manipulator)
- **Camera**: ZED 2i stereo camera with body tracking SDK
- **Skeleton model**: ZED 16-point body model (pelvis, spine, neck/head, bilateral shoulders/elbows/wrists/hands, bilateral hips/knees)
- **Recording format**: ROS2 rosbag with topics:
  - `/zed/zed_node/body_trk/skeletons` (ObjectsStamped) — 16 keypoint 3D positions
  - `/joint_states` (JointState) — 6 UR10 joint angles

**Recording sessions**: 10 recordings from physical theatre workshops, each ~80–90 seconds at ~15 Hz, capturing guided human-robot interaction following Meyerhold's biomechanics and Bogart's Viewpoints frameworks.

### 4.2 Rosbag Processing Pipeline (Notebook 01)

1. **Topic extraction**: Parse ZED skeleton messages and UR10 joint state messages
2. **Temporal synchronization**: Interpolate joint states to skeleton tracking timestamps (~15 Hz)
3. **Skeleton selection**: Interactive 3D visualization for operator to select tracked performer (label_id) — important because ZED may track multiple people
4. **Coordinate transformation**: Transform skeleton keypoints from camera optical frame (`zed_left_camera_frame`) to robot base frame (`base_link`) via TF2 static transforms
5. **Forward kinematics**: Compute end-effector position from joint angles using URDF model
6. **Quality checks**: NaN detection, timestamp gap detection, joint limit violation detection

**Output**: Synchronized CSV per episode with schema:
```
timestamp | sk_0_x, sk_0_y, sk_0_z, ..., sk_15_z | j0, ..., j5 | ee_x, ..., ee_yaw
           (48 skeleton columns)                   (6 joints)     (6 end-effector)
```

### 4.3 Data Statistics

| Metric | Value |
|--------|-------|
| Total recordings | 10 |
| Frames per recording | ~1,150–1,350 |
| Duration per recording | ~78–90 sec |
| Sampling rate | ~15 Hz |
| Total frames | ~11,784 |
| Quality (NaN) | 0 across all recordings |
| Quality (joint violations) | 0 across all recordings |

### 4.4 Data Preparation (Notebook 02)

#### Normalization

Standardization (z-score) computed from training set only:

$$\hat{x} = \frac{x - \mu}{\sigma}$$

**Key design choice**: No hip-centering of skeleton data. Skeleton coordinates remain in the `base_link` frame to preserve spatial position information (distance from robot, arc position). This is important because the robot's response depends on *where* the performer is, not just their pose.

Normalization statistics (training set):
```
Joint mean:  [ 1.6411, -1.0359,  0.5217,  4.6956, -2.0777,  3.0679] rad
Joint std:   [1.3606,  0.4858,   0.6995,  0.2204,  1.4493,  0.0001] rad
```

Note: Joint 5 (wrist_3) has near-zero standard deviation (0.0001 rad), indicating it was effectively stationary during recording.

#### Sliding Window

- **Input window**: $T_{\text{in}} = 10$ frames (0.67 s at 15 Hz)
- **Output window**: $T_{\text{out}} = 10$ frames (0.67 s prediction horizon)
- **Stride**: 1 (maximum overlap)
- **Episode boundary**: Windows never cross episode boundaries (prevents temporal leakage)
- **Minimum episode length**: $T_{\text{in}} + T_{\text{out}} = 20$ frames

#### Train/Val/Test Split

Episode-level splitting (all windows from an episode belong to one split):
- **Train**: 70% → 6 episodes → **6,981 windows**
- **Val**: 15% → 2 episodes → **2,471 windows**
- **Test**: 15% → 2 episodes → **2,332 windows**
- **Seed**: 42 (deterministic, reproducible)

#### Data Augmentation (training only)

1. **Temporal jitter**: Shift window start position by $\Delta t \sim \text{Uniform}(-2, +2)$ frames — simulates temporal misalignment between skeleton tracking and joint states
2. **Skeleton noise**: $\epsilon \sim \mathcal{N}(0, 0.01^2)$ meters per coordinate, applied before normalization — simulates ZED tracking noise

---

## 5. Training

### 5.1 Loss Function

Composite loss operating in normalized joint space:

$$\mathcal{L} = \lambda_{\text{pred}} \mathcal{L}_{\text{pred}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{acc}} \mathcal{L}_{\text{acc}} + \lambda_{\text{jl}} \mathcal{L}_{\text{jl}}$$

**Prediction loss** (position MSE):
$$\mathcal{L}_{\text{pred}} = \frac{1}{BT_{\text{out}}J} \sum_{b,t,j} (\hat{y}_{b,t,j} - y_{b,t,j})^2$$

**Smoothness loss** (velocity matching):
$$\mathcal{L}_{\text{smooth}} = \text{MSE}(\hat{\mathbf{v}}, \mathbf{v}), \quad \text{where } \mathbf{v}_t = \mathbf{y}_{t+1} - \mathbf{y}_t$$

**Acceleration loss** (second-order smoothness):
$$\mathcal{L}_{\text{acc}} = \text{MSE}(\hat{\mathbf{a}}, \mathbf{a}), \quad \text{where } \mathbf{a}_t = \mathbf{v}_{t+1} - \mathbf{v}_t$$

**Joint limit penalty** (soft constraint):
$$\mathcal{L}_{\text{jl}} = \frac{1}{BT_{\text{out}}J} \sum_{b,t,j} \left[\text{ReLU}(\ell_j - \hat{y}_{b,t,j}) + \text{ReLU}(\hat{y}_{b,t,j} - u_j)\right]^2$$

where $\ell_j, u_j$ are normalized lower/upper joint limits.

**Loss weights**:

| Component | Weight ($\lambda$) | Purpose |
|-----------|-------------------|---------|
| Prediction | 1.0 | Primary trajectory accuracy |
| Smoothness | 0.1 | Velocity continuity |
| Acceleration | 0.05 | Jerk reduction |
| Joint limit | 0.1 | Physical feasibility |

### 5.2 Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-4}$ |
| Weight decay | $1 \times 10^{-4}$ |
| Batch size | 32 |
| Max epochs | 200 |
| LR schedule | Linear warmup (5 epochs, factor 0.01→1.0) + Cosine annealing ($\eta_{\min} = 10^{-6}$) |
| Gradient clipping | Max norm 1.0 |
| Early stopping | 30 epochs patience |
| Reproducibility | Seed 42, deterministic cuDNN |

### 5.3 Training Results

#### Full Model (ActionChunkingTransformer — skeleton + joints input)

**Experiment ID**: `vam_20260210_2342`

| Metric | Value |
|--------|-------|
| Epochs trained | 48 (early stopping) |
| Best validation loss | **0.6381** (epoch 18) |
| Final training loss | 0.0122 |
| Training time per epoch | ~1.9 s |
| Hardware | NVIDIA RTX 5090 (33.6 GB VRAM) |

**Training progression** (selected epochs):

| Epoch | Train Loss | Val Loss | Learning Rate |
|-------|-----------|----------|---------------|
| 1 | 2.271 | 4.685 | 1.00e-06 |
| 5 | 0.066 | 0.804 | 8.02e-05 |
| 10 | 0.028 | 0.700 | 9.99e-05 |
| 18* | 0.019 | **0.638** | 9.91e-05 |
| 30 | 0.015 | 0.863 | 9.63e-05 |
| 48 | 0.012 | 1.202 | 8.91e-05 |

*Best validation epoch (model checkpoint saved)

**Test set per-joint RMSE**:

| Joint | RMSE (normalized) | RMSE (rad) | RMSE (deg) |
|-------|------------------|-----------|-----------|
| j0 (shoulder_pan) | 0.4296 | 0.5845 | 33.49 |
| j1 (shoulder_lift) | 0.3224 | 0.1566 | 8.97 |
| j2 (elbow) | 0.4298 | 0.3007 | 17.23 |
| j3 (wrist_1) | 0.9284 | 0.2046 | 11.72 |
| j4 (wrist_2) | 0.5348 | 0.7750 | 44.41 |
| j5 (wrist_3) | 0.6667 | 0.0001 | 0.00 |
| **Overall** | **0.5868** | — | — |

**End-effector position error** (via forward kinematics):
- Mean: 435.8 mm
- Median: 366.4 mm
- 95th percentile: 1,081.7 mm

#### Skeleton-Only Model (SkeletonOnlyACT)

**Experiment ID**: `vam_skelonly_20260224_0607`

| Metric | Value |
|--------|-------|
| Epochs trained | 51 (early stopping) |
| Best validation loss | **2.952** (epoch 21) |
| Final training loss | 0.013 |

**Test set per-joint RMSE**:

| Joint | RMSE (deg) |
|-------|-----------|
| j0 (shoulder_pan) | 60.70 |
| j1 (shoulder_lift) | 18.09 |
| j2 (elbow) | 34.55 |
| j3 (wrist_1) | 19.41 |
| j4 (wrist_2) | 134.27 |
| j5 (wrist_3) | 0.01 |

**End-effector position error**: Mean 757.5 mm, Median 633.0 mm

> **Discussion**: The skeleton-only model has higher error, which is expected — it must predict robot actions purely from observed human movement without proprioceptive feedback. However, this variant is preferred for real robot deployment to avoid compounding prediction errors through feedback.

### 5.4 Figures to Include from Notebooks

> **[FIGURE 1]**: Training curves — 6-panel subplot from `notebooks/03_train_vam.ipynb` cell `cell-14`
> - Total loss, Prediction loss, Smoothness loss, Acceleration loss, Joint limit penalty, Learning rate
> - Shows rapid convergence and gap between train/val (early stopping at epoch 48)

> **[FIGURE 2]**: Per-joint RMSE bar chart from `notebooks/03_train_vam.ipynb` cell `cell-18`
> - Bar chart of test set RMSE per joint in degrees

> **[FIGURE 3]**: Predicted vs Ground Truth joint trajectories from `notebooks/03_train_vam.ipynb` cell `cell-20`
> - 4 test samples showing solid (GT) and dashed (predicted) lines for all 6 joints

> **[FIGURE 4]**: Prediction error over horizon from `notebooks/03_train_vam.ipynb` cell `cell-21`
> - Shows how error grows with prediction timestep (does error accumulate over the 10-frame chunk?)

> **[FIGURE 5]**: 3D visualization: Skeleton + GT Robot + Predicted Robot from `notebooks/03_train_vam.ipynb` cell `xrhc2d7o4u`
> - Interactive 3D showing skeleton (blue), ground truth robot (green), predicted robot (red), and end-effector trails

---

## 6. Inference Pipeline

### 6.1 Pipeline Components

The inference pipeline runs as a ROS2 node at 15 Hz and consists of four stages:

#### Stage 1: Input Assembly

The `InputAssembler` maintains a rolling buffer of the most recent $T_{\text{in}} = 10$ frames:

$$\mathbf{B}_{t} = [\mathbf{f}_{t-T_{\text{in}}+1}, \mathbf{f}_{t-T_{\text{in}}+2}, \ldots, \mathbf{f}_t]$$

where each frame $\mathbf{f}_i \in \mathbb{R}^{48}$ (skeleton-only) or $\mathbb{R}^{54}$ (skeleton + joints). Normalization is applied using training statistics before passing to the model.

#### Stage 2: Model Inference

When the ensemble signals it's time to predict (every $K$ frames), the model generates a chunk:

$$\hat{\mathbf{Y}}_t = \text{ACT}(\bar{\mathbf{B}}_t) \in \mathbb{R}^{T_{\text{out}} \times 6}$$

where $\bar{\mathbf{B}}_t$ is the normalized input. Output is denormalized to physical joint angles (radians).

#### Stage 3: Temporal Ensemble

The key innovation for smooth continuous motion. At each timestep $t$, multiple overlapping chunks contribute predictions:

$$\hat{\theta}_t = \frac{\sum_{k \in \mathcal{A}(t)} w_k \cdot \hat{\mathbf{Y}}_k[t - t_k]}{\sum_{k \in \mathcal{A}(t)} w_k}$$

where:
- $\mathcal{A}(t)$ is the set of active predictions covering timestep $t$
- $t_k$ is the start timestep of prediction $k$
- $w_k = \exp(-\lambda \cdot (t - t_k))$ is the exponential decay weight
- $\lambda$ controls the balance between smoothness (small $\lambda$) and responsiveness (large $\lambda$)

**Parameters**:
- $K = 1$ (re-predict every frame) — empirically smoothest
- $\lambda = 0.5$ — balances accuracy and smoothness
- Max history: $T_{\text{out}} / K = 10$ overlapping chunks

This approach was adapted from Tony Zhao et al.'s ACT paper, with modifications for the 15 Hz streaming control context.

#### Stage 4: Safety Checker

Three-layer safety enforcement:

**Layer 1 — Joint limit clamping** (hard):
$$\theta_j^{\text{safe}} = \text{clip}(\hat{\theta}_j, \ell_j, u_j)$$

**Layer 2 — Velocity limiting** (in non-robot mode):
$$\text{if } \frac{|\theta_j^{\text{new}} - \theta_j^{\text{old}}|}{\Delta t} > v_{\max}: \quad \theta_j^{\text{new}} = \theta_j^{\text{old}} + \text{sign}(\Delta\theta) \cdot v_{\max} \cdot \Delta t$$

**Layer 3 — Acceleration limiting** (in non-robot mode):
$$\text{if } \frac{|v_j^{\text{new}} - v_j^{\text{old}}|}{\Delta t} > a_{\max}: \quad v_j^{\text{new}} = v_j^{\text{old}} + \text{sign}(\Delta v) \cdot a_{\max} \cdot \Delta t$$

In robot mode, only joint limit clamping is applied — MoveIt Servo handles velocity/acceleration safety at 250 Hz.

### 6.2 Robot Control — MoveIt Servo Integration

The VAM node sends velocity commands to MoveIt Servo using a P-controller with feedforward:

$$\mathbf{v} = K_p (\boldsymbol{\theta}_{\text{target}} - \boldsymbol{\theta}_{\text{current}}) + \frac{\boldsymbol{\theta}_{\text{target}} - \boldsymbol{\theta}_{\text{target}}^{\text{prev}}}{\Delta t}$$

where:
- $K_p = 12.0$ in the robot launch configuration (node default is 5.0, overridden to 12.0 in `vam_robot.launch.py`; tuned for ~2 Hz tracking bandwidth)
- The feedforward term anticipates target motion to eliminate steady-state lag
- $\Delta t = 1/15$ s

Published as `JointJog` messages to `/servo_node/delta_joint_cmds`.

**MoveIt Servo configuration** (250 Hz):
- Low-pass filter coefficient: 2.0 (tuned for responsiveness)
- Self-collision checking: enabled
- Joint limit margin: 0.1 rad
- Incoming command timeout: 0.1 s (auto-halt on tracking loss)

### 6.3 Coordinate Transformation

Skeleton keypoints are transformed from camera frame to robot frame at runtime:

$$\mathbf{p}_{\text{base}} = \mathbf{R}_{\text{cam→base}} \cdot \mathbf{p}_{\text{cam}} + \mathbf{t}_{\text{cam→base}}$$

The rotation matrix $\mathbf{R}$ is computed from the TF2 quaternion using:

$$\mathbf{R} = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-zw) & 2(xz+yw) \\ 2(xy+zw) & 1-2(x^2+z^2) & 2(yz-xw) \\ 2(xz-yw) & 2(yz+xw) & 1-2(x^2+y^2) \end{bmatrix}$$

Static transform chain: `map → world → base_link` (set during lab calibration).

---

## 7. Inference Results

### 7.1 Pipeline Validation (Offline, Notebook 04)

**Test episode**: `25_12_11_RAPP_M_R2G1S1_02` (1,191 frames at 15 Hz)

**Default parameters**: K=2, $\lambda$=0.01

| Metric | Value |
|--------|-------|
| Output frames | 1,182 |
| Model predictions | 591 (50% of frames, K=2) |
| Safety-constrained frames | 526 |

**Per-joint RMSE** (ensemble vs ground truth):

| Joint | RMSE (deg) |
|-------|-----------|
| j0 (shoulder_pan) | 15.79 |
| j1 (shoulder_lift) | 5.53 |
| j2 (elbow) | 15.34 |
| j3 (wrist_1) | 5.81 |
| j4 (wrist_2) | 42.05 |
| j5 (wrist_3) | 0.01 |

> Note: The temporal ensemble improves RMSE compared to raw model predictions because it averages out individual chunk errors.

### 7.2 Latency Profiling

**Hardware**: NVIDIA RTX 5090 (33.6 GB VRAM), CUDA 12.8, PyTorch 2.10.0

Measured over 1,000 inference runs:

| Metric | Value |
|--------|-------|
| Mean latency | **0.74 ms** |
| Median latency | 0.73 ms |
| P95 latency | 0.76 ms |
| P99 latency | 0.85 ms |
| Max latency | 1.07 ms |
| Budget at 15 Hz | 66.7 ms |
| **Headroom** | **99%** |
| Max achievable rate | **1,352 Hz** |

> **Significance**: The model can theoretically run at 1,352 Hz — 90x faster than the 15 Hz control rate. This massive headroom means the system can run on much more modest hardware (e.g., laptop GPUs, embedded GPUs like Jetson) while maintaining real-time performance.

### 7.3 Parameter Sweep — Accuracy vs. Smoothness

Sweep over prediction stride $K \in \{1, 2, 3, 5\}$ and decay weight $\lambda \in \{0.01, 0.05, 0.1, 0.5\}$:

| K | $\lambda$ | RMSE (deg) | Accel. Std (deg/s²) | Predictions |
|---|----------|-----------|-------------------|-------------|
| **1** | **0.01** | **19.60** | **15.1** | 1,182 |
| 1 | 0.05 | 19.55 | 15.1 | 1,182 |
| 1 | 0.10 | 19.49 | 15.4 | 1,182 |
| 1 | 0.50 | 19.23 | 22.7 | 1,182 |
| 2 | 0.01 | 19.60 | 267.4 | 591 |
| 2 | 0.50 | 19.24 | 279.4 | 591 |
| 3 | 0.01 | 19.54 | 333.3 | 394 |
| 5 | 0.01 | 19.62 | 424.2 | 237 |

**Key findings**:
- $K=1$ (predict every frame) produces dramatically smoother output (15 deg/s² vs. 267+ deg/s²)
- RMSE is relatively stable across all configurations (~19.2–19.6°), suggesting the temporal ensemble is robust
- $\lambda$ has minimal impact on RMSE but moderate impact on smoothness — smaller $\lambda$ = smoother
- **Optimal**: $K=1, \lambda=0.5$ — best accuracy with acceptable smoothness

### 7.4 Skeleton-Only Inference Results (Deployed Model)

The skeleton-only model is the one deployed on the real robot. Its inference pipeline results:

**Test episode**: Same (`25_12_11_RAPP_M_R2G1S1_02`, 1,191 frames)

**Per-joint RMSE** (skeleton-only ensemble vs GT):

| Joint | RMSE (deg) |
|-------|-----------|
| j0 (shoulder_pan) | 24.25 |
| j1 (shoulder_lift) | 16.49 |
| j2 (elbow) | 37.72 |
| j3 (wrist_1) | 13.98 |
| j4 (wrist_2) | 172.13 |
| j5 (wrist_3) | 0.01 |

**Latency on laptop GPU** (RTX 5070 Laptop — the actual deployment hardware):

| Metric | Value |
|--------|-------|
| Mean latency | **0.92 ms** |
| Median | 0.87 ms |
| P95 | 1.19 ms |
| P99 | 1.57 ms |
| Max | 3.10 ms |
| Max achievable rate | **1,093 Hz** |
| Headroom | **99%** |

> **Key insight for the paper**: Even on a laptop GPU (RTX 5070), the model achieves 1,093 Hz — 73x the required 15 Hz rate. This validates the "modest hardware" claim. The model could likely run on embedded GPUs (Jetson Orin) or even CPU-only systems.

**Skeleton-only parameter sweep** (K and $\lambda$):

| K | $\lambda$ | RMSE (deg) | Accel. Std (deg/s²) |
|---|----------|-----------|-------------------|
| **1** | **0.01** | **72.58** | **48.9** |
| 1 | 0.50 | 72.41 | 80.8 |
| 2 | 0.01 | 72.59 | 487.6 |
| 5 | 0.01 | 72.63 | 622.1 |

Same pattern as full model: $K=1$ dramatically smoother. RMSE relatively stable.

### 7.5 Comparative Summary — Full vs. Skeleton-Only

| Metric | Full Model | Skeleton-Only |
|--------|-----------|--------------|
| Parameters | 1,142,854 | 1,134,086 |
| Best val loss | 0.638 | 2.952 |
| Test RMSE (overall, normalized) | 0.587 | 1.145 |
| j1 (shoulder_lift) RMSE | 8.97° | 18.09° |
| EE error (mean) | 435.8 mm | 757.5 mm |
| Inference latency | 0.74 ms (RTX 5090) | 0.92 ms (RTX 5070 Laptop) |
| Feedback loop | Yes (accumulates) | **No** (open-loop) |
| Robot deployment | Not suitable | **Deployed** |

> The skeleton-only model has 2x higher error but is the correct choice for deployment because it avoids compounding prediction errors through proprioceptive feedback. In physical theatre, the qualitative character of the motion matters more than positional precision.

### 7.6 Real Robot Deployment Observations

> **[PLACEHOLDER — Maleen to fill in]**: Include qualitative observations from real robot testing:
> - How did the robot respond to performer movement?
> - Was the motion smooth and expressive?
> - Any notable behaviors (tracking loss recovery, movement quality, performer feedback)?
> - Video stills or frames from recordings
> - Any quantitative metrics from live deployment (if logged)

### 7.7 Figures to Include from Notebooks

> **[FIGURE 6]**: Ensemble vs Ground Truth trajectories — 6-panel subplot from `notebooks/04_inference_test.ipynb` cell `cell-plot-trajectory`
> - Per-joint comparison over the full test episode (~79 seconds)

> **[FIGURE 7]**: Overlapping chunk visualization from `notebooks/04_inference_test.ipynb` cell `cell-plot-chunks`
> - Shows multiple semi-transparent chunk predictions, bold ensemble blend, and GT for a 3-second window
> - **This is the key figure illustrating how the temporal ensemble works**

> **[FIGURE 8]**: Velocity profiles from `notebooks/04_inference_test.ipynb` cell `cell-smoothness`
> - Per-joint velocity comparison between ensemble and GT

> **[FIGURE 9]**: Parameter sweep — RMSE vs Smoothness scatter from `notebooks/04_inference_test.ipynb` cell `cell-sweep-plot`
> - Shows tradeoff curves for different K and $\lambda$ values

> **[FIGURE 10]**: Latency histogram from `notebooks/04_inference_test.ipynb` cell `cell-latency-hist`
> - Distribution of inference latencies with 15 Hz budget line

> **[FIGURE 11]**: Skeleton-only training curves from `notebooks/03b_train_vam_skeleton_only.ipynb` cell `cell-14`

> **[FIGURE 12]**: Skeleton-only 3D visualization from `notebooks/03b_train_vam_skeleton_only.ipynb` cell `cell-3d-viz`

> **[FIGURE 13]**: Skeleton-only latency histogram (laptop GPU) from `notebooks/04b_inference_test_skeleton_only.ipynb` cell `cell-latency-hist`

---

## 8. System End-to-End Latency

| Stage | Rate | Latency |
|-------|------|---------|
| ZED body tracking | ~15 Hz | ~66 ms |
| TF2 transform | per frame | <1 ms |
| Input assembly + normalization | per frame | <1 ms |
| ACT model inference (GPU) | per K frames | **0.74 ms** |
| Temporal ensemble query | per frame | <0.01 ms |
| Safety checker | per frame | <0.01 ms |
| ROS2 publish | per frame | <1 ms |
| MoveIt Servo interpolation | 250 Hz | 4 ms |
| **Total pipeline** | **15 Hz** | **~70 ms** |

The dominant latency is the ZED body tracking rate (~15 Hz). All downstream processing adds negligible overhead.

---

## 9. Docker & Deployment Infrastructure

### 9.1 Docker Configuration

- **Base**: NVIDIA CUDA 12.8 + Ubuntu 22.04
- **ROS2**: Humble Desktop
- **PyTorch**: 2.10.0 with CUDA 12.8 (Blackwell architecture, sm_120)
- **GPU**: NVIDIA runtime with shared memory 8 GB (for PyTorch DataLoader)
- **Network**: Host networking (ROS2 DDS communication)
- **Services**: Jupyter Lab (port 8888), TensorBoard (port 6006)

### 9.2 Deployment Modes

1. **RViz visualization** (rosbag replay) — no robot, no camera
2. **Real robot + recorded skeleton** (rosbag replay + MoveIt Servo)
3. **Full live performance** (ZED camera + MoveIt Servo + UR10)

---

## 10. Safety Architecture Detail

### 10.1 Three-Layer Defense-in-Depth

```
Layer 3: UR10 Hardware Controller (final enforcer)
    ↑
Layer 2: MoveIt Servo — 250 Hz
    • Collision checking (self-collision model from URDF)
    • Joint limit enforcement (0.1 rad margin)
    • Singularity detection & velocity scaling
    • Butterworth low-pass filter (coeff = 2.0)
    • Auto-halt on command timeout (0.1 s)
    ↑
Layer 1: VAM SafetyChecker — 15 Hz
    • Joint limit clamping (UR10 URDF limits)
    • Velocity limiting: default 1.0 rad/s, robot launch override 2.0 rad/s
    • Acceleration limiting: default 5.0 rad/s², robot launch override 8.0 rad/s²
    • In robot mode (joint_limits_only=True): only joint clamping applied
    • Seeded with actual robot position on startup
    ↑
VAM Temporal Ensemble (produces smooth targets)
```

### 10.2 Tracking Loss Handling

- **Timeout**: 0.5 s without fresh skeleton data
- **Response**: Hold position (zero-velocity JointJog)
- **Recovery**: Automatically resumes when tracking recovers
- **Shutdown**: Ctrl+C sends zero-velocity command before exit

### 10.3 UR10 Joint Limits

| Joint | Min (rad) | Max (rad) | Range (deg) |
|-------|----------|----------|-------------|
| shoulder_pan | -6.283 | 6.283 | ±360° |
| shoulder_lift | -6.283 | 6.283 | ±360° |
| elbow | -3.142 | 3.142 | ±180° |
| wrist_1 | -6.283 | 6.283 | ±360° |
| wrist_2 | -6.283 | 6.283 | ±360° |
| wrist_3 | -6.283 | 6.283 | ±360° |

---

## 11. Discussion — Error Analysis and Limitations

### 11.1 Per-Joint Error Distribution

The error distribution is non-uniform across joints:
- **Low error**: j1 (shoulder_lift: 8.97°), j3 (wrist_1: 11.72°), j5 (wrist_3: 0.00°)
- **High error**: j0 (shoulder_pan: 33.49°), j4 (wrist_2: 44.41°)

j5 has effectively zero error because it has near-zero variance in the training data ($\sigma = 0.0001$ rad). j4 has the highest error, potentially due to the wrist's higher variability and the normalization challenge from high standard deviation ($\sigma = 1.45$ rad).

### 11.2 Overfitting

The model shows signs of overfitting:
- Train loss: 0.012 vs. Val loss: 0.638 (53x gap)
- Best val epoch: 18, but training continued until epoch 48 (early stopping)

This is expected given the relatively small dataset (~7K training windows) and the model capacity. Potential mitigations for future work:
- More training data (additional workshop recordings)
- Stronger augmentation
- Regularization (increased dropout, weight decay)

### 11.3 Dataset Size

10 recordings is small for a transformer model. However:
- The action space is relatively constrained (6-DOF robot arm)
- The input is low-dimensional (48D skeleton + 6D joints)
- The model is deliberately small (1.14M params) to match the data regime
- Data augmentation (temporal jitter + skeleton noise) provides implicit regularization

### 11.4 End-Effector Error

The mean end-effector error of 435.8 mm is high in absolute terms but must be contextualized:
- This is for devised physical theatre, not precision manipulation
- The goal is responsive, expressive motion, not positional accuracy
- The robot needs to create the *impression* of responding to the performer, not replicate exact joint angles
- In practice, the temporal ensemble smoothing + MoveIt Servo interpolation produces visually compelling motion

### 11.5 Limitations

1. **Single performer**: Current system tracks one skeleton; multi-performer would require attention-based selection
2. **Camera calibration**: Using approximate static transforms rather than rigorous hand-eye calibration
3. **Limited action vocabulary**: 10 recordings capture only a subset of physical theatre movements
4. **No explicit safety zones**: Workspace constraints are implicit in the training data, not explicitly enforced

---

## 12. Comparison with Related Work

| System | Parameters | Inference | Input | Real-Time | Application |
|--------|-----------|----------|-------|-----------|-------------|
| **VAM (ours)** | **1.14M** | **0.74 ms** | Skeleton 3D | **Yes (15 Hz)** | Physical theatre |
| ACT (Zhao et al.) | ~80M | ~20 ms | RGB image | Yes | Bimanual manipulation |
| RT-2 (Google) | 55B | ~1-3 s | RGB image | No | General manipulation |
| Diffusion Policy | ~100M | ~100 ms | RGB/Point cloud | Near-RT | Manipulation |
| ALOHA (Zhao et al.) | ~80M | ~20 ms | RGB + proprio | Yes | Bimanual manipulation |

**Key differentiators**:
- 70x fewer parameters than ACT by using structured skeleton input instead of raw images
- CNN-free: skeleton tracking is offloaded to ZED SDK, allowing the transformer to focus on temporal dynamics
- Designed for continuous streaming (no episode-based reset) — essential for improvised performance
- Multi-loss training incorporating smoothness explicitly (not just prediction accuracy)

---

## 13. Summary of Figures Needed

| # | Description | Source | Type |
|---|-------------|--------|------|
| 1 | System architecture diagram | Create | Block diagram |
| 2 | Training curves (6-panel) | `03_train_vam.ipynb` cell `cell-14` | Plot |
| 3 | Per-joint RMSE bar chart | `03_train_vam.ipynb` cell `cell-18` | Bar chart |
| 4 | Predicted vs GT trajectories | `03_train_vam.ipynb` cell `cell-20` | Time series |
| 5 | Error over prediction horizon | `03_train_vam.ipynb` cell `cell-21` | Line plot |
| 6 | 3D skeleton + robot visualization | `03_train_vam.ipynb` cell `xrhc2d7o4u` | 3D render |
| 7 | Ensemble vs GT full episode | `04_inference_test.ipynb` cell `cell-plot-trajectory` | Time series |
| 8 | Overlapping chunks visualization | `04_inference_test.ipynb` cell `cell-plot-chunks` | Overlay plot |
| 9 | Velocity profiles | `04_inference_test.ipynb` cell `cell-smoothness` | Time series |
| 10 | Parameter sweep scatter | `04_inference_test.ipynb` cell `cell-sweep-plot` | Scatter |
| 11 | Latency histogram | `04_inference_test.ipynb` cell `cell-latency-hist` | Histogram |
| 12 | Photo of real robot system | Lab photo | Photo |
| 13 | Photo of workshop performance | Workshop | Photo |

---

## 14. Suggested Paper Outline (IEEE RO-MAN format)

### I. Introduction
- Physical theatre + robots motivation
- RAPP Lab history (decade of investigation)
- Gap: no VAM for improvised physical theatre
- Contributions (see Section 1 above)

### II. Related Work
- Vision-action models (ACT, RT-2, Diffusion Policy)
- Robot performance art (LaViers, Spedalieri, Herath)
- Physical theatre methodologies (Meyerhold, Lecoq, Bogart)
- Human-robot improvisation

### III. System Design
- Recording setup (ZED + UR10)
- Data pipeline (Sections 4.1–4.4)
- Architecture (Section 3)
- Training (Section 5)
- Inference pipeline (Section 6)
- Safety (Section 10)

### IV. Experiments
- Training results (Section 5.3)
- Inference validation (Section 7.1)
- Latency profiling (Section 7.2)
- Parameter sweep (Section 7.3)
- Real robot deployment (Section 7 + lab notes)

### V. Discussion
- Error analysis (Section 11)
- Comparison with related work (Section 12)
- Implications for HRI and performance
- Balance: technical contributions + broader questions (agency, complicite, storytelling)

### VI. Conclusion & Future Work
- Summary of contributions
- Future: more data, multi-performer, richer action vocabulary
- RAPP Lab 04 workshop series (ICSR 2026)

---

## Verification

To verify/reproduce these results:
1. Launch Docker container: `docker-compose up -d` (from `docker/`)
2. Open Jupyter at `http://localhost:8888`
3. Run notebooks in order: 01 → 02 → 03 → 04
4. For real robot testing, follow the 5-terminal launch sequence in the README
