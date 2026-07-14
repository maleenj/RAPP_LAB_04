"""Action Chunking Transformer for Vision-Action Model.

Architecture:
    Input [B, T_in, 54] → Split skeleton/robot
    → Skeleton Projection (Linear) + Robot Projection (MLP)
    → Additive Fusion + Positional Encoding
    → TransformerEncoder (context sequence)
    → Learned Action Queries + TransformerDecoder (cross-attention)
    → Output Projection → [B, T_out, 6]
"""

import math

import torch
import torch.nn as nn

from vam_utils.config.model_config import ModelConfig


class ActionChunkingTransformer(nn.Module):
    """Action Chunking Transformer for predicting robot joint trajectories
    from skeleton + robot state history."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Input projections ---
        self.skeleton_proj = nn.Linear(config.skeleton_dim, config.d_model)

        self.robot_proj = nn.Sequential(
            nn.Linear(config.joint_dim, config.d_robot_hidden),
            nn.GELU(),
            nn.Linear(config.d_robot_hidden, config.d_model),
        )

        # --- Positional encoding (learned) ---
        self.encoder_pos = nn.Embedding(config.T_in, config.d_model)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # --- Action queries (learned, one per output timestep) ---
        self.action_queries = nn.Embedding(config.T_out, config.d_model)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # --- Output projection ---
        self.output_proj = nn.Linear(config.d_model, config.joint_dim)

        # --- Initialize weights ---
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Embeddings: small normal
        nn.init.normal_(self.encoder_pos.weight, std=0.02)
        nn.init.normal_(self.action_queries.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T_in, input_dim] where input_dim = skeleton_dim + joint_dim

        Returns:
            [batch, T_out, joint_dim] predicted future joint angles
        """
        B = x.shape[0]

        # Split input into skeleton and robot components
        skeleton = x[..., :self.config.skeleton_dim]   # [B, T_in, 48]
        robot = x[..., self.config.skeleton_dim:]       # [B, T_in, 6]

        # Project to d_model
        skel_emb = self.skeleton_proj(skeleton)          # [B, T_in, d_model]
        robot_emb = self.robot_proj(robot)               # [B, T_in, d_model]

        # Additive fusion
        fused = skel_emb + robot_emb                     # [B, T_in, d_model]

        # Add positional encoding
        positions = torch.arange(self.config.T_in, device=x.device)
        fused = fused + self.encoder_pos(positions)      # broadcast over batch

        # Encode
        memory = self.encoder(fused)                     # [B, T_in, d_model]

        # Decode with learned action queries
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        # queries: [B, T_out, d_model]

        decoded = self.decoder(queries, memory)          # [B, T_out, d_model]

        # Project to joint space
        output = self.output_proj(decoded)               # [B, T_out, joint_dim]

        return output

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SkeletonOnlyACT(nn.Module):
    """Action Chunking Transformer — skeleton-only input variant.

    Identical to ActionChunkingTransformer except:
    - No robot_proj MLP (no joint angles in input)
    - No additive fusion — skeleton embedding goes directly to encoder
    - Input: [B, T_in, skeleton_dim=48] instead of [B, T_in, 54]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.skeleton_proj = nn.Linear(config.skeleton_dim, config.d_model)
        self.encoder_pos = nn.Embedding(config.T_in, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        self.action_queries = nn.Embedding(config.T_out, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        self.output_proj = nn.Linear(config.d_model, config.joint_dim)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, normal for embeddings."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.encoder_pos.weight, std=0.02)
        nn.init.normal_(self.action_queries.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, T_in, skeleton_dim] — skeleton data only (48-dim)

        Returns:
            [batch, T_out, joint_dim] predicted future joint angles
        """
        B = x.shape[0]
        skel_emb = self.skeleton_proj(x)
        positions = torch.arange(self.config.T_in, device=x.device)
        skel_emb = skel_emb + self.encoder_pos(positions)
        memory = self.encoder(skel_emb)
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.output_proj(decoded)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
