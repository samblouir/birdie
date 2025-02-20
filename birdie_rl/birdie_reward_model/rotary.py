# This is copied from Llama 3's codebase and is modified to:
# 1. Fix bugs with compile
# 2. Make it easier to update the cached rotary position encodings in loaded models
"""
rotary.py

PURPOSE:
	- Provides methods for computing rotary embeddings (rotary positional encodings),
	  commonly used in certain transformer architectures.
	- The apply_rotary_emb function applies the rotation to Q,K in attention.

NOTE:
	- This code references "apply_scaling" for additional transformations,
	  which can be toggled with use_scaled.
"""

import math
from typing import Optional, Tuple
import torch


def apply_scaling(freqs: torch.Tensor, old_context_len: int) -> torch.Tensor:
	"""
	Demonstration function to scale frequencies for extended context lengths.
	The logic used is a sample from certain research prototypes.

	Args:
		freqs (Tensor): The base frequency tensor.
		old_context_len (int): A reference context length to decide scaling.

	Returns:
		Tensor: The scaled frequencies.
	"""
	scale_factor = 8
	low_freq_factor = 1
	high_freq_factor = 4

	low_freq_wavelen = old_context_len / low_freq_factor
	high_freq_wavelen = old_context_len / high_freq_factor
	new_freqs = []
	for freq in freqs:
		wavelen = 2 * math.pi / freq
		if wavelen < high_freq_wavelen:
			new_freqs.append(freq)
		elif wavelen > low_freq_wavelen:
			new_freqs.append(freq / scale_factor)
		else:
			# Weighted blend region
			smooth = (old_context_len / wavelen - low_freq_factor) / (
				high_freq_factor - low_freq_factor
			)
			new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

	return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
	dim: int,
	end: int,
	theta: float = 500000.0,
	use_scaled: bool = False,
	old_context_len: int = 32768,
	accelerator=None,
):
	"""
	Precompute the complex representation of rotary frequencies.

	Args:
		dim (int): The dimension, typically half of the hidden dim for cos/sin pairs.
		end (int): The maximum length or sequence size for which to compute.
		theta (float): A base scale factor (e.g., 1e5).
		use_scaled (bool): Whether to apply the custom 'apply_scaling' function.
		old_context_len (int): Reference length for scaling.
		accelerator: Optional accelerate object for printing debugging (unused).

	Returns:
		Tensor: shape [end, dim], containing the cis() representation of frequencies.
	"""
	# Base frequencies
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
	t = torch.arange(end, device=freqs.device, dtype=torch.float32)

	# If scaling is enabled
	if use_scaled:
		freqs = apply_scaling(freqs, old_context_len=old_context_len)

	# Outer product to get shape [end, dim//2]
	freqs = torch.outer(t, freqs)
	# Convert to complex
	freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
	return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
	"""
	Reshape the precomputed freqs_cis to broadcast with x, which might be [batch, seq, heads, dim].
	This is a simplified version that expects x.shape[1]==freqs_cis.shape[0] and x.shape[-1]==freqs_cis.shape[1].
	"""
	ndim = x.ndim
	shape = [1 if (i != 1 and i != ndim - 1) else d for i, d in enumerate(x.shape)]
	shape[1] = freqs_cis.shape[0]
	shape[-1] = freqs_cis.shape[1]
	return freqs_cis.view(*shape).to(x.device)


@torch._dynamo.disable
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
	"""
	Apply the rotary embedding to Q,K by interpreting them as complex numbers.

	Args:
		xq (Tensor): shape [batch, seq, heads, dim].
		xk (Tensor): same shape as xq.
		freqs_cis (Tensor): shape [seq, dim], precomputed.

	Returns:
		(Tensor, Tensor): The rotated Q,K.
	"""
	xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
	xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

	freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
	xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
	xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
	return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
	"""
	Utility to expand key/value heads if needed.
	"""
	if n_rep == 1:
		return x
	bs, slen, n_kv_heads, head_dim = x.shape
	return (
		x[:, :, :, None, :]
		.expand(bs, slen, n_kv_heads, n_rep, head_dim)
		.reshape(bs, slen, n_kv_heads * n_rep, head_dim)
	)
