# basemodel.py
"""
===============================================================================
===============================================================================
!!! FULL VERSION COMING SOON WITH BIRDIE_DNA !!!
===============================================================================
===============================================================================
BASE MODEL
-------------------------------------
Supports:
  - Rotary embeddings
  - GQA (Group Query Attention) (set (gqa_num_heads < num_heads) and (gqa_num_heads % num_heads == 0))
  - RMSNorm
  - (Optional) fused cross-entropy that does not materialize logits
  - Segment-aware block mask
  - FAN-in (as seen in JAX, similar to OLMO 2) param inits
===============================================================================
"""


from birdie_dna.modeling import rotary, softcap
from birdie_dna.modeling.rope_embedding import fast_rope_embedding
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.checkpoint import checkpoint
from typing import Optional, Any
import birdie_dna.utils
import einops
import math
import os
import torch
import torch.nn as nn
import torch.nn.attention.flex_attention as flex_attention
import torch.nn.functional as F


################################################################################
# RMSNorm
################################################################################
	
class RMSNorm(nn.Module):
	"""
	Root Mean Square Layer Normalization.
	"""
	def __init__(
		self,
		hidden_size: int,
		eps: float = 1e-5,
		dtype: torch.dtype = torch.float32,
		device: Optional[str] = None,
		**kwargs
	):
		super().__init__()
		torch_dtype = birdie_dna.utils.str_to_dtype(dtype)
		self.norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=True, dtype=torch_dtype, device=device)
		

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return self.norm(x)



################################################################################
# Tanh Softcap
################################################################################
# tanh_softcap = softcap.generate_tanh_softcap(soft_cap=50, approx=True)
# tanh_softcap_last_layer = softcap.generate_tanh_softcap(soft_cap=30, approx=True)


################################################################################
# LinearProjection
################################################################################

class LinearProjection(nn.Module):
	"""
	A single linear layer with forced dimension alignment (dims are made to be divisible by 128)
	Uses truncated normal initialization for weights.
	"""
	def __init__(self, in_dim: int = None, out_dim: int = None, **kwargs: Any):
		super().__init__()

		# Grab input/output dims
		if in_dim is None:
			in_dim = kwargs["hidden_size"]
		if out_dim is None:
			out_dim = kwargs.get("out_dim", in_dim)

		is_vocab_head = kwargs.get("is_vocab_head", False)
		vocab_size = kwargs.get("vocab_size", 32000)
		if is_vocab_head:
			out_dim = vocab_size

		param_dtype = kwargs.get("dtype", torch.float32)

		# Make in_dim and out_dim multiples of 128 for performance
		in_dim = birdie_dna.utils.make_divisible_by(in_dim, 128)
		out_dim = birdie_dna.utils.make_divisible_by(out_dim, 128)

		param_dtype = birdie_dna.utils.str_to_dtype(param_dtype)

		# Build linear layer
		self.layer = nn.Linear(
			in_dim,
			out_dim,
			bias=kwargs.get("projection_layers_use_bias", False),
			dtype=param_dtype,
		)

		# Truncated normal init
		# fan_in = out_dim
		fan_in = in_dim
		std = 1.0 / math.sqrt(fan_in)
		nn.init.trunc_normal_(
			self.layer.weight,
			mean=0.0,
			std=std,
			a=-2 * std,
			b=2 * std
		)

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return self.layer(x)


################################################################################
# MHA (Multi-Head Attention) using flex_attention
################################################################################
class MHA(nn.Module):
	"""
	Custom MHA that:
	  - Splits Q,K,V
	  - Applies flex_attention
	  - Optionally uses post-attention RMSNorm
	  - Applies rotary embeddings if provided
	  - Supports GQA via gqa_num_heads
	"""
	def __init__(self, **kwargs):
		super().__init__()
		self.hidden_size = kwargs["hidden_size"]
		self.num_heads   = kwargs["num_heads"]
		self.head_dim    = kwargs.get("head_dim", self.hidden_size // self.num_heads)
		self.gqa_num_heads = int(kwargs.get("gqa_num_heads", self.num_heads))

		# freqs_cis = kwargs.get("freqs_cis", None)
		# self.register_buffer("freqs_cis", freqs_cis, persistent=False,)

		# Q/K dimension for Q,K
		qk_dims = self.num_heads * self.head_dim
		# For GQA, V can differ
		v_dims = self.gqa_num_heads * self.head_dim

		# Q,K,V projections
		self.q_proj = LinearProjection(self.hidden_size, qk_dims, **kwargs)
		self.k_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)
		self.v_proj = LinearProjection(self.hidden_size, v_dims, **kwargs)

		# Final out projection
		self.o_proj = LinearProjection(qk_dims, self.hidden_size, **kwargs)
		# Post-RMSNorm
		self.post_rms_norm = RMSNorm(
			hidden_size=self.hidden_size,
			eps=kwargs.get("eps", 1e-5),
			dtype=kwargs.get("dtype", torch.float32),
			device=kwargs.get("device", None),
		)

		self.enable_gqa = (self.gqa_num_heads != self.num_heads)
		if self.enable_gqa:
			assert(self.gqa_num_heads % self.num_heads == 0), "gqa_num_heads must be a multiple of num_heads"
			assert(self.gqa_num_heads < self.num_heads), "gqa_num_heads must be less than num_heads"


	@torch.compile
	def forward(self, x: torch.Tensor, block_mask=None, freqs_cis=None, *args, **kwargs,) -> torch.Tensor:
		"""
		Forward pass for the MHA block.
		"""
		residual = x

		x = self.post_rms_norm(x)

		# Project Q,K,V
		q = self.q_proj(x)
		k = self.k_proj(x)
		v = self.v_proj(x)


		q = einops.rearrange(q, "B S (H D) -> B S H D", H=self.num_heads)
		k = einops.rearrange(k, "B S (H D) -> B S H D", H=self.gqa_num_heads)
		q, k = fast_rope_embedding(q, k, freqs_cis.real, freqs_cis.imag)
		q = einops.rearrange(q, "B S H D -> B H S D", H=self.num_heads)
		k = einops.rearrange(k, "B S H D -> B H S D", H=self.gqa_num_heads)
		v = einops.rearrange(v, "B S (H D) -> B H S D", H=self.gqa_num_heads)
		# flex_attention
		attn_out = flex_attention.flex_attention(
			query=q,
			key=k,
			value=v,
			block_mask=block_mask,
			enable_gqa=self.enable_gqa,
		)

		# Reshape back
		attn_out = einops.rearrange(attn_out, "B H S D -> B S (H D)", H=self.num_heads)

		# Output projection
		out = self.o_proj(attn_out)

		# Residual
		return (residual + out).to(x.dtype)


################################################################################
# SwiGLU Feed-Forward
################################################################################

class SwiGLU(nn.Module):
	"""
	A feed-forward block using the 'SwiGLU' pattern:
	  - gate = sigmoid(Linear(x))
	  - ungated = Linear(x)
	  - multiply gate * ungated
	  - project down
	  - RMSNorm
	  - add residual
	"""
	def __init__(self, **kwargs):
		super().__init__()
		hidden_size = kwargs["hidden_size"]
		mlp_mult = kwargs.get("mlp_dim_mult", 4.0)

		# Round hidden_size to multiple of 16 for HPC alignment if desired
		in_dim = birdie_dna.utils.make_divisible_by(hidden_size, 128)
		ffn_dim = birdie_dna.utils.make_divisible_by(int(in_dim * mlp_mult), 128)

		# Two parallel input layers
		self.wi_0 = LinearProjection(in_dim, ffn_dim, **kwargs)
		self.wi_1 = LinearProjection(in_dim, ffn_dim, **kwargs)

		# Output projection
		self.wo = LinearProjection(ffn_dim, in_dim, **kwargs)

		# RMSNorm
		self.rms_norm = RMSNorm(
			hidden_size=in_dim,
			eps=kwargs.get("eps", 1e-5),
			dtype=kwargs.get("dtype", torch.float32),
			device=kwargs.get("device", None),
		)

	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		"""
		Forward pass of SwiGLU feed-forward.
		"""
		residual = x
		x = self.rms_norm(x)
		gated = torch.sigmoid(self.wi_0(x))
		ungated = self.wi_1(x)
		ff_out = self.wo(gated * ungated)
		return ff_out + residual
	
################################################################################
# Embedding wrapper
################################################################################
class Embedding(nn.Embedding):
	"""
	Wrapper to allow for *args and **kwargs.
	"""
	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return super().forward(x)


################################################################################
# Dropout wrapper
################################################################################
class Dropout(nn.Dropout):
	"""
	Wrapper to allow for *args and **kwargs.
	"""
	def forward(self, x: torch.Tensor, *args, **kwargs,) -> torch.Tensor:
		return super().forward(x)

################################################################################
# BaseModel
################################################################################


class BaseModel(nn.Module):
	"""
	A flexible Transformer-like model that:
	  1) Has an embedding layer (vocab_size x hidden_size).
	  2) Stacks MHA + MLP layers (with optional RMSNorm, GQA, rotary, etc.).
	  3) Ends with a final RMSNorm, and has a projection to vocab_size (assuming we're doing LM).

	If label_ids is provided, returns cross-entropy loss. Otherwise returns logits.
	"""
	def __init__(self, layer_kwargs):
		super().__init__()

		# Basic config
		self.num_layers      = layer_kwargs["num_layers"]
		self.hidden_size     = layer_kwargs.get("hidden_size", 2048)
		self.vocab_size      = layer_kwargs.get("vocab_size", 32000)
		self.sequence_length = layer_kwargs.get("sequence_length", 512)
		self.batch_size      = layer_kwargs.get("batch_size", 1)

		self.num_heads = layer_kwargs["num_heads"]
		self.head_dim  = layer_kwargs.get("head_dim", self.hidden_size // self.num_heads)

		self.use_precomputed_block_mask = int(layer_kwargs.get("use_precomputed_block_mask", 0))
		self.use_fusedlce = int(layer_kwargs.get("use_fusedlce", 0))
		self.bidirectional = int(layer_kwargs.get('bidirectional', 0))


		# Embedding
		self.embeddings = Embedding(self.vocab_size, self.hidden_size)
		fan_in = self.hidden_size
		std = 1.0 / math.sqrt(fan_in)
		nn.init.trunc_normal_(
			self.embeddings.weight,
			mean=0.0,
			std=std,
			a=-2 * std,
			b=2 * std
		)

		# Precompute rotary embeddings
		freqs_cis = rotary.precompute_freqs_cis(
			dim=(self.head_dim),
			end=self.sequence_length,
			theta=layer_kwargs.get("base_decay_rate", 500_000),
			use_scaled=False,
			old_context_length=layer_kwargs.get("pretraining_sequence_length", self.sequence_length)
		)
		# register buffer
		self.register_buffer("freqs_cis", freqs_cis, persistent=False,)

		embed_dropout = layer_kwargs.get("embed_dropout", 0.0)
		residual_dropout = layer_kwargs.get("residual_dropout", 0.0)

		# Build sub-layers
		layers = []
		seen_layers = 0
		while seen_layers < self.num_layers:
			if layer_kwargs.get("use_attention", True):
				mha = MHA(
					**layer_kwargs,
					freqs_cis=self.freqs_cis,
				)
				layers.append(mha)
				seen_layers += 1
				if (0.0 < residual_dropout): layers.append(Dropout(p=residual_dropout, inplace=True))

			if layer_kwargs.get("use_mlp", True):
				ffn = SwiGLU(**layer_kwargs)
				layers.append(ffn)
				seen_layers += 1
				if (0.0 < residual_dropout): layers.append(Dropout(p=residual_dropout, inplace=True))

		# Final RMSNorm
		layers.append(
			RMSNorm(
				hidden_size=self.hidden_size,
				eps=layer_kwargs.get("eps", 1e-5),
				dtype=layer_kwargs.get("dtype", torch.float32),
				device=layer_kwargs.get("device", None),
			)
		)

		# Vocab head
		head_in_dim = birdie_dna.utils.make_divisible_by(self.hidden_size, 128)
		head_out_dim = self.vocab_size
		self.vocab_head = nn.Parameter(torch.randn(head_in_dim, head_out_dim), requires_grad=True)
		fan_in_head = head_out_dim
		std_head = 1.0 / math.sqrt(fan_in_head)
		nn.init.trunc_normal_(
			self.vocab_head,
			mean=0.0,
			std=std_head,
			a=-2 * std_head,
			b=2 * std_head
		)

		# Optionally import fused LCE
		if self.use_fusedlce:
			from cut_cross_entropy import LinearCrossEntropy
			self.LCE = LinearCrossEntropy()

		# Construct layers
		self.layers = nn.ModuleList()
		self.layers.append(self.embeddings)  # first is embedding
		if (0.0 < embed_dropout): self.layers.append(Dropout(p=embed_dropout, inplace=True))
		self.layers.extend(layers)

		# Possibly build block_mask once
		if self.use_precomputed_block_mask:
			def mask_mod(b, h, q_idx, kv_idx):
				# Strictly causal
				return (q_idx >= kv_idx)
			self.block_mask = create_block_mask(
				mask_mod,
				B=self.batch_size,
				H=1,
				Q_LEN=self.sequence_length,
				KV_LEN=self.sequence_length,
				device=layer_kwargs.get("device", "cuda"),
				_compile=True,
				# BLOCK_SIZE=128,
			)
		else:
			self.block_mask = None

		# Simple cross-entropy per sample
		def cross_entropy_per_sample(logits, label_ids):
			"""
			logits: (B, L, vocab_size)
			label_ids: (B, L) with -100 to ignore
			Returns per-sample average, shape (B,)
			"""
			logits_t = logits.permute(0, 2, 1)  # -> (B, vocab_size, L)
			label_ids_ = label_ids.to(torch.long)
			loss_per_pos = F.cross_entropy(logits_t, label_ids_, reduction='none')
			mask = (label_ids_ != -100)
			sum_loss = (loss_per_pos * mask).sum(dim=1)
			count = mask.sum(dim=1).clamp(min=1)
			return sum_loss / count

		self.cross_entropy_per_sample = cross_entropy_per_sample


	def update_freqs_cis(self, freqs_cis):
		"""
		Updates the freqs_cis buffer in the model and all MHA layers.
		TODO: I believe I removed the stored buffers in all layers, using a single shared buffer.
		May need to return to an old approach for model layers split across accelerators.
		"""
		try:
			self.register_buffer("freqs_cis", freqs_cis, persistent=False,)
		except RuntimeError:
			self.freqs_cis = freqs_cis
		print(f"  Updated model.freqs_cis.shape to {self.freqs_cis.shape}")

		for layer in self.layers:
			if isinstance(layer, MHA):
				try:
					layer.register_buffer("freqs_cis", freqs_cis, persistent=False,)
				except RuntimeError:
					layer.freqs_cis = freqs_cis

	def reset_freqs_cis(self, seq_len: int, base_decay_rate: float = 500_000, old_context_length: int = None, accelerator=None,):
		"""
		Recompute the rotary embeddings for a new fixed sequence length.
		"""
		if old_context_length is None:
			old_context_length = seq_len

		self.sequence_length = seq_len
		freqs_cis = rotary.precompute_freqs_cis(
			dim=(self.head_dim),
			end=seq_len,
			theta=base_decay_rate,
			use_scaled=(old_context_length < seq_len),
			old_context_length=old_context_length,
			accelerator=accelerator,
		)
		return self.update_freqs_cis(freqs_cis)

	def forward(
		self,
		input_ids: torch.Tensor,
		label_ids: Optional[torch.Tensor] = None,
		segment_ids: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		return_per_sample_loss: bool = False,
		**kwargs
	) -> torch.Tensor:
		"""
		Forward pass. If label_ids are provided, return scalar cross-entropy. Otherwise, logits.
		"""
		B, L = input_ids.shape
		if segment_ids is None:
			segment_ids = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)

		if self.bidirectional:
			def mask_mod(b, h, q_idx, kv_idx):
				# Check if both query and key/value tokens are in the prefix (i.e., attention_mask is 1)
				prefix_q = (attention_mask[b, q_idx] == 1)
				prefix_kv = (attention_mask[b, kv_idx] == 1)
				prefix_mask = (prefix_q & prefix_kv)

				# Causal mask: allow attention only to previous tokens (including current token)
				causal_mask = q_idx >= kv_idx

				# Combine prefix mask and causal mask using logical OR
				prefix_or_causal_mask = prefix_mask | causal_mask

				# Segment mask: allow attention only within the same segment
				segment_mask = segment_ids[b, q_idx] == segment_ids[b, kv_idx]

				# Final mask: combine all masks using logical AND
				return prefix_or_causal_mask & segment_mask
		else:

			def mask_mod(b, h, q_idx, kv_idx):
				# Causal mask with support for segment_ids
				causal_mask = (q_idx >= kv_idx)
				segment_mask = (segment_ids[b, q_idx] == segment_ids[b, kv_idx])
				return causal_mask & segment_mask

		block_mask = create_block_mask(
			mask_mod,
			B=B,
			H=1,
			Q_LEN=L,
			KV_LEN=L,
			device=input_ids.device,
			_compile=True
		)

		# Pass through the layers
		x = input_ids
		for layer in self.layers:
			x = layer(x, block_mask=block_mask, freqs_cis=self.freqs_cis)

		# If label_ids were not provided, return logits
		if label_ids is None:
			B, L, D = x.shape
			logits = torch.matmul(x.view(-1, D), self.vocab_head.to(x.dtype))
			logits = logits.view(B, L, self.vocab_size)
			return logits

		# Else compute cross-entropy
		# 1) If fused LCE is enabled:
		if self.use_fusedlce:
			logits_16 = x.to(torch.float16)
			w_16 = self.vocab_head.transpose(0, 1).to(torch.float16)
			loss = self.LCE(logits_16, w_16, label_ids)
			if return_per_sample_loss:
				return loss
			return loss.mean()
		
		# 2) Otherwise standard cross-entropy:
		x = x.to(torch.bfloat16)
		logits = torch.matmul(x.view(-1, x.shape[-1]), self.vocab_head.to(x.dtype))
		logits = logits.view(B, L, self.vocab_size)
		per_sample_loss = self.cross_entropy_per_sample(logits, label_ids)
		if return_per_sample_loss:
			return per_sample_loss
		return per_sample_loss.mean()
