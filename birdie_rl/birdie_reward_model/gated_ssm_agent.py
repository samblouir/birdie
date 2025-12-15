"""
gated_ssm_agent.py

PURPOSE:
  - Contains building blocks for an MLPModel that can also incorporate attention
    layers, RMSNorm, rotary embeddings, and a custom gating structure. 

USAGE:
  from birdie_rl.birdie_reward_model import gated_ssm_agent
  model = gated_ssm_agent.MLPModel(...)
  output = model(x, current_seq_len=some_len)

CONTENTS:
  1) RMSNorm, RMS_split: Normalization layers
  2) SwiGLU: Activation block
  3) MHA: Multi-Head Attention with block_mask
  4) GatedSSM: (unused by default, but provided for advanced gating)
  5) MLPModel: Main model combining RMS_split, MHA, and SwiGLU layers
"""

import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

# Rotary embeddings for Q,K
from birdie_rl.birdie_reward_model import rotary
try:
    from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention
    _HAS_FLEX_ATTENTION = True
except Exception:
    _create_block_mask = None
    _flex_attention = None
    _HAS_FLEX_ATTENTION = False

if _HAS_FLEX_ATTENTION and hasattr(torch, "compile"):
    try:
        create_block_mask = torch.compile(_create_block_mask)
    except Exception:
        create_block_mask = _create_block_mask
else:
    create_block_mask = _create_block_mask

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")
# Some convenience references
softmax = nn.Softmax(dim=-1)
sigmoid = nn.Sigmoid()

# Default maximum sequence length for dimension alignment in MLPModel
default_max_seq_len = 2048


class RMSNorm(nn.Module):
    """
    A root mean square layer normalization (RMSNorm).
    Normalizes across the last dimension by dividing by the RMS of the values.

    Attributes:
      eps (float): A small epsilon to avoid division by zero.
      scale (Parameter): A learnable scale parameter, shape [dims].
    """
    def __init__(self, dims, scale_init_fn=torch.ones, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(scale_init_fn(dims))

    def forward(self, x):
        """
        Forward pass: normalize by RMS along the last dimension, then multiply by scale.
        """
        # (x.shape[-1] ** -0.5) is sqrt(1/dims)
        norm = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.scale * x / (norm + self.eps)


class RMS_split(nn.Module):
    """
    A variant of RMSNorm that can optionally re-project the dimension after normalization,
    and includes an optional dropout.

    Usage:
      block = RMS_split(input_dim=128, output_dim=256, dropout_rate=0.1)
      y = block(x)

    Attributes:
      norm (RMSNorm): The RMSNorm sub-module.
      output_dim (int or None): If set, a linear layer to re-project to output_dim.
      dropout (nn.Dropout): The dropout layer (rate=dropout_rate).
      scale (nn.Parameter): Additional learnable scale multiplier.
    """
    def __init__(self, input_dim, output_dim=None, dropout_rate=0.0, eps=1e-8):
        super(RMS_split, self).__init__()
        self.norm = RMSNorm(dims=input_dim, eps=eps)
        self.scale = nn.Parameter(torch.zeros(input_dim) + 0.1)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        if self.output_dim is not None:
            self.out = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        """
        Normalize x via RMSNorm, scale it, optionally project to output_dim.
        """
        x = self.norm(x)
        x = x * self.scale
        if self.output_dim is None:
            return x
        return self.out(x)


class SwiGLU(nn.Module):
    """
    SwiGLU block:
      - 2x projection input -> (main, gate)
      - Non-linear gating with sigmoid(gate)
      - Another projection back to residual dimension
      - Contains an RMS_split to normalize input

    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dims, hidden_dims):
        super(SwiGLU, self).__init__()
        dims = int(dims)
        hidden_dims = int(hidden_dims)

        self.input_dims = dims
        self.hidden_dims = hidden_dims

        # Projects from dims to hidden_dims*2 (split into main + gate)
        self.wi = nn.Linear(dims, hidden_dims * 2, bias=False)
        # Projects back to dims
        self.wo = nn.Linear(hidden_dims, dims, bias=False)

        # RMS-split normalizes the input
        self.norm = RMS_split(dims)

    def forward(self, x):
        """
        Forward pass:
          1) RMS normalize
          2) wi -> (main, gate)
          3) main * sigmoid(gate)
          4) wo
          5) Add residual
        """
        residual = x
        x = self.norm(x)
        x = self.wi(x)
        (main, gate) = x.chunk(2, dim=-1)
        x = main * torch.sigmoid(gate)
        return self.wo(x) + residual


class MHA(nn.Module):
    """
    Multi-Head Attention block with potential GQA (grouped query heads),
    and usage of rotary embeddings for Q,K.

    Specifically references "torch.nn.attention.flex_attention.flex_attention"
    for the final attention operation, using an optional block_mask for 
    causal or custom attention patterns.
    """
    def __init__(
        self,
        dims,
        head_dims=64,
        freqs_cis=None,
    ):
        """
        Args:
          dims (int): The model dimension.
          head_dims (int): The dimension per head.
          freqs_cis (Tensor or None): Precomputed rotary embeddings.
        """
        super(MHA, self).__init__()

        # For demonstration: the number of heads for Q can differ from K/V if GQA is used
        self.num_heads = dims // head_dims
        self.gqa_num_heads = 2  # e.g., grouping for keys/values
        if self.num_heads < self.gqa_num_heads or (self.num_heads % self.gqa_num_heads) != 0:
            self.gqa_num_heads = 1

        # Dimensions for Q, K, V
        q_dims = head_dims * self.num_heads
        v_dims = head_dims * self.gqa_num_heads

        self.freqs_cis = freqs_cis

        # Linear projections
        self.q_proj = nn.Linear(dims, q_dims, bias=False)
        self.k_proj = nn.Linear(dims, v_dims, bias=False)
        self.v_proj = nn.Linear(dims, v_dims, bias=False)
        self.o_proj = nn.Linear(q_dims, dims, bias=False)

        # RMS-split for input norm
        self.norm = RMS_split(dims)

    def forward(self, x, block_mask=None, attn_mask=None):
        """
        Forward pass for MHA with optional block_mask.

        Args:
          x (Tensor): shape [batch, seq_len, dims]
          block_mask (Tensor or None): shape e.g. [1,1,Q_LEN,KV_LEN], 
                                       for controlling which positions can attend.
          attn_mask (Tensor or None): bool or float attention mask broadcastable to
                                      [batch, heads, Q_LEN, KV_LEN] (used for SDPA fallback).
        Returns:
          Tensor: shape [batch, seq_len, dims]
        """
        residual = x
        x = self.norm(x)

        # Project Q,K with different #heads if using GQA
        q = einops.rearrange(self.q_proj(x), 'b s (h d) -> b s h d', h=self.num_heads)
        k = einops.rearrange(self.k_proj(x), 'b s (h d) -> b s h d', h=self.gqa_num_heads)

        # Apply rotary embeddings to Q,K if freqs_cis is provided
        if self.freqs_cis is not None:
            q, k = rotary.apply_rotary_emb(q, k, self.freqs_cis)

        # Reshape Q,K for 'flex_attention': [b, heads, seq, dims]
        q = einops.rearrange(q, 'b s h d -> b h s d')
        k = einops.rearrange(k, 'b s h d -> b h s d')

        # Project V similarly, shape [b, gqa_num_heads, seq, dims]
        v = einops.rearrange(self.v_proj(x), 'b s (h d) -> b h s d', h=self.gqa_num_heads)

        use_flex_attention = _HAS_FLEX_ATTENTION and (block_mask is not None) and (_flex_attention is not None)
        if use_flex_attention:
            # Use the custom "flex_attention" function from PyTorch's experimental module
            mha_out = _flex_attention(
                query=q,
                key=k,
                value=v,
                block_mask=block_mask,
                enable_gqa=(self.gqa_num_heads != self.num_heads),
            )
        else:
            # Fallback to SDPA (CPU-safe, available in stable PyTorch).
            if k.shape[1] != q.shape[1]:
                if (q.shape[1] % k.shape[1]) != 0:
                    raise ValueError(
                        f"GQA head mismatch: q_heads={q.shape[1]} not divisible by kv_heads={k.shape[1]}"
                    )
                repeat_factor = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)
            mha_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        # Rearrange back
        mha_out = einops.rearrange(mha_out, 'b h s d -> b s (h d)')

        # Final projection to match dims
        mha_out = self.o_proj(mha_out)
        return residual + mha_out


class GatedSSM(nn.Module):
    """
    Gated State Space Model block (for advanced usage).
    Not typically used directly in MLPModel but kept for reference.

    The idea:
      - Project input to a state dims (K).
      - Another projection that splits into (u, g_in, g_out).
      - Recursively combine states across the sequence dimension.
    """
    def __init__(self, dims, state_size_mult=2, dropout_rate=0.0):
        super(GatedSSM, self).__init__()
        self.norm = RMS_split(dims)
        state_dims = dims * state_size_mult
        self.K_proj = nn.Linear(dims, state_dims, bias=False)
        self.ugg_proj = nn.Linear(dims, state_dims * 3, bias=False)
        self.out_proj = nn.Linear(state_dims, dims, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass for Gated SSM. 
        Note: This is a conceptual example, not typically used in the MLPModel.

        Args:
          x (Tensor): shape [batch, seq_len, dims]

        Returns:
          Tensor: shape [batch, seq_len, dims]
        """
        residual = x
        x = self.norm(x)

        # gating
        K = torch.sigmoid(self.K_proj(x))
        u, g_in, g_out = self.ugg_proj(x).chunk(3, dim=-1)
        u = u * torch.sigmoid(g_in) * (1 - K)
        g_out = torch.sigmoid(g_out)

        outputs = [u[..., 0:1, :]]
        # Recursively apply
        for idx in range(1, K.shape[-2]):
            A0 = K[..., idx - 1 : idx, :]
            B0 = outputs[-1]
            B1 = u[..., idx : idx + 1, :]
            outputs.append(A0 * B0 + B1)

        outputs = torch.cat(outputs, dim=-2)
        outputs = outputs * g_out
        y = self.out_proj(outputs)
        return y + residual


class MLPModel(nn.Module):
    """
    An MLP-like model that can incorporate multiple MHA+SwiGLU layers
    with RMS normalization and optional block_mask usage.

    Typical usage in e.g. a RL/bandit setting to predict next-step reward or improvement.

    Args:
      input_dim (int): Dimension of the input features.
      output_dim (int): Dimension of the output features.
      hidden_dims (list): Each entry is a layer dimension.
      dropout_rate (float): Dropout rate for RMS-split or layers.
      num_heads (int): For attention blocks if used.
      max_seq_len (int): For precomputing rotary embeddings or block length.
      device: CPU or GPU device for loading the embeddings.

    The forward pass calls a custom 'layer_fn' that constructs a block_mask for attention
    (e.g. a causal mask) if desired.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[64, 64],
        dropout_rate=0.0,
        num_heads=1,
        max_seq_len=default_max_seq_len,
        device=None,
    ):
        super(MLPModel, self).__init__()

        # Store layers
        self.layers = nn.ModuleList()

        # Decide a head dimension for MHA
        # If the last hidden dim is available, use that; else default 64
        if len(hidden_dims) >= 2:
            self.head_dims = hidden_dims[-2] // 4
        else:
            self.head_dims = 64

        # Precompute rotary embeddings if desired
        self.freqs_cis = rotary.precompute_freqs_cis(
            self.head_dims,
            max_seq_len,
            use_scaled=(max_seq_len != default_max_seq_len),
        ).to(device) if device else None

        self.sequence_length = max_seq_len

        # RMS-split from input_dim to hidden_dims[0]
        self.layers.append(RMS_split(input_dim, hidden_dims[0], dropout_rate=dropout_rate))

        # Build the hidden layers with MHA+SwiGLU interleaving
        for i in range(len(hidden_dims)):
            # Insert MHA
            self.layers.append(
                MHA(
                    dims=hidden_dims[i],
                    head_dims=self.head_dims,
                    freqs_cis=self.freqs_cis,
                )
            )
            # Insert SwiGLU
            self.layers.append(SwiGLU(hidden_dims[i], hidden_dims[i] * (8/3)))

        # Finally, an RMS-split and a linear for the output
        self.layers.append(RMS_split(hidden_dims[-1]))
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=False))

        self.cuda_device = device

        # A function to handle forward pass with optional block_mask creation
        def layer_fn(
            x,
            current_seq_len=None,
            attention_mask=None,
            reset_mask=None,
            segment_ids=None,
            **kwargs,
        ):
            """
            A specialized forward function that can create a block_mask for e.g. causal attention.

            Args:
              x (Tensor): shape [batch, seq_len, features]
              current_seq_len (int): The actual sequence length in usage.
              attention_mask (Tensor or None): Possibly [batch, seq_len].
              reset_mask, segment_ids: Additional flags (unused here).
              **kwargs: Extra arguments.

            Returns:
              Tensor: The final output after all layers, shape [batch, seq_len, output_dim].
            """

            # Example: create a block_mask for causal or user-defined constraints
            # define a function to check if Q_idx >= K_idx, etc.
            def mask_mod(b, h, q_idx, kv_idx):
                """
                Simple causal example:
                  - allow attention to previous tokens only
                  - clamp at current_seq_len if we are using a smaller actual length
                """
                is_valid_loc = (q_idx < current_seq_len) & (kv_idx < current_seq_len)
                causal_mask = (q_idx >= kv_idx)
                return causal_mask & is_valid_loc

            block_mask = None
            attn_mask = None

            if current_seq_len is None:
                current_seq_len = x.shape[1]
            current_seq_len = int(current_seq_len)

            if create_block_mask is not None:
                # Build the block_mask using create_block_mask if available.
                block_mask = create_block_mask(
                    mask_mod,
                    B=x.shape[0],  # or 1 if we want same mask for entire batch
                    H=1,  # or num_heads
                    Q_LEN=x.shape[1],
                    KV_LEN=x.shape[1],
                    device=x.device,
                )
            else:
                # SDPA fallback mask:
                # - causal for tokens < current_seq_len
                # - ensure rows for padded queries still have at least one valid entry
                #   to avoid NaNs (we allow self-attend on padded positions).
                seq_len = x.shape[1]
                q_idx = torch.arange(seq_len, device=x.device)[:, None]
                kv_idx = torch.arange(seq_len, device=x.device)[None, :]
                causal = q_idx >= kv_idx

                if current_seq_len >= seq_len:
                    attn_mask = causal
                else:
                    valid_q = q_idx < current_seq_len
                    valid_kv = kv_idx < current_seq_len
                    diag = q_idx == kv_idx
                    attn_mask = (valid_q & valid_kv & causal) | (~valid_q & diag)

            # Pass through each layer
            for layer in self.layers:
                if isinstance(layer, MHA):
                    x = layer(x, block_mask=block_mask, attn_mask=attn_mask)
                else:
                    x = layer(x)

            return torch.tanh(x)

        self.layer_fn = layer_fn

    def forward(self, x, **kwargs):
        """
        Standard forward entrypoint. Delegates to self.layer_fn.

        Args:
          x (Tensor): shape [batch, seq_len, input_dim]
          **kwargs: might include current_seq_len for partial usage.

        Returns:
          Tensor: shape [batch, seq_len, output_dim]
        """
        output = self.layer_fn(x, **kwargs)
        return output
