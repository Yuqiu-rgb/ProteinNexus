import torch
import torch.nn as nn
import torch.nn.functional as F


class DyNexFiLMFusion(nn.Module):
    """
    DyNex-FiLM: A Spatially-Dynamic, Nexus-Conditioned Fusion Module.

    This module enhances the Hyper-FiLM concept by introducing position-aware
    dynamic modulation. It fuses two input tensors by generating unique FiLM
    parameters for each position in the sequence, allowing the fusion strategy
    to adapt to local context.

    Args:
        dim (int): The feature dimension of the input tensors (D).
        context_proj_dim (int, optional): The projection dimension for the context
            encoder. Defaults to 256.
        hypernet_hidden_dim (int, optional): The hidden dimension of the hypernetwork
            MLP. Defaults to 512.
    """

    def __init__(self, dim: int, context_proj_dim: int = 256, hypernet_hidden_dim: int = 512):
        super().__init__()
        self.dim = dim

        # 1. Enhanced Context Encoder
        # Input is [x_seq, x_struct, x_seq * x_struct], hence dim * 3
        self.context_encoder = nn.Sequential(
            nn.Linear(dim * 3, context_proj_dim),
            nn.ReLU(inplace=True)
        )

        # 2. Hypernetwork (operates on positional context)
        self.hypernetwork = nn.Sequential(
            nn.Linear(context_proj_dim, hypernet_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_hidden_dim, dim * 4)  # Output for 2 gammas and 2 betas per position
        )

        # 3. Gated Aggregation
        # Input to the gate will be [x_seq, x_struct, x'_seq, x'_struct]
        self.gate_linear = nn.Linear(dim * 4, 1)

    def forward(self, x_seq: torch.Tensor, x_struct: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DyNexFiLMFusion module.

        Args:
            x_seq (torch.Tensor): The sequence embeddings tensor of shape.
            x_struct (torch.Tensor): The structure embeddings tensor of shape.

        Returns:
            torch.Tensor: The fused tensor of shape.
        """
        B, L, D = x_seq.shape
        assert x_struct.shape == (B, L, D), "Input tensors must have the same shape"
        assert D == self.dim, f"Input dimension {D} does not match module dimension {self.dim}"

        # 1. Enhanced Context Encoder with pre-fusion interaction
        x_interact = x_seq * x_struct
        x_concat = torch.cat([x_seq, x_struct, x_interact], dim=-1)  # Shape:

        # Generate position-aware context. No global pooling.
        context_positional = self.context_encoder(x_concat)  # Shape:

        # 2. Hypernetwork generates position-wise FiLM parameters
        # params shape:
        params = self.hypernetwork(context_positional)

        # Split params into gammas and betas for both modalities
        # Each will have shape
        gamma_seq, beta_seq, gamma_struct, beta_struct = torch.split(params, self.dim, dim=-1)

        # 3. Spatially-Dynamic FiLM Modulation
        # Apply FiLM transformation directly (no broadcasting needed)
        x_seq_prime = gamma_seq * x_seq + beta_seq
        x_struct_prime = gamma_struct * x_struct + beta_struct

        # 4. Final Gated Aggregation
        # Prepare input for the gate
        gate_input = torch.cat([x_seq, x_struct, x_seq_prime, x_struct_prime], dim=-1)  # Shape:
        # Compute position-wise gate
        gate = torch.sigmoid(self.gate_linear(gate_input))  # Shape:

        # Apply gate to fuse the modulated features
        fused_output = gate * x_seq_prime + (1 - gate) * x_struct_prime

        return fused_output