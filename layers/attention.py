import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
import utils
from typing import Optional
import layers.easydel_flash as flash
import functools

class MultiheadedAttention(nnx.Module):
	'''
	Multi-headed attention mechanism.
	'''

	def __init__(
			self,
			d_input: int,
			rngs: int,
			fully_causal: bool = False,
			head_size: Optional[int] = None,
			use_bias: bool = False,
			param_dtype: str = 'float32',
			compute_dtype: str = 'float32',
			attn_fn: Optional[str] = None,
			# attn_fn: Optional[str] = 'easydel_triton',
			**kwargs,
		):
		self.d_input = d_input
		self.rngs = rngs
		self.fully_causal = fully_causal
		self.head_size = head_size
		self.num_heads = (d_input // self.head_size)
		self.use_bias = use_bias
		self.param_dtype = param_dtype
		self.compute_dtype = compute_dtype
		self.attn_fn = attn_fn

		assert (self.head_size * self.num_heads == d_input), "Head size times num heads must equal d_input."

		# Additional keyword arguments are captured in `kwargs`.
		for key, value in kwargs.items():
			setattr(self, key, value)

		param_dtype = utils.str_to_jax_dtype(param_dtype)
		compute_dtype = utils.str_to_jax_dtype(compute_dtype)

		if self.attn_fn in ["triton", "easydel_triton"]:
			self.attn_dpa_fn = flash.run_mhdpa
			self.attn_bias_fn = flash.calculate_bias
		else:
			self.attn_dpa_fn = functools.partial(
				nnx.dot_product_attention,
				dtype=compute_dtype,
			)

		shared_kwargs = dict(
			dtype=compute_dtype,
			param_dtype=param_dtype,
			rngs=self.rngs,
		)

		self.norm = nnx.RMSNorm(
			num_features=d_input,
			**shared_kwargs,
		)

		key_features = self.num_heads * self.head_size

		self.w_query = nnx.Linear(
			in_features=d_input,
			out_features=key_features,
			use_bias=use_bias,
			**shared_kwargs,
		)
		self.w_key = nnx.Linear(
			in_features=d_input,
			out_features=key_features,
			use_bias=use_bias,
			**shared_kwargs,
		)
		self.w_value = nnx.Linear(
			in_features=d_input,
			out_features=key_features,
			use_bias=use_bias,
			**shared_kwargs,
		)
		self.w_output = nnx.Linear(
			in_features=key_features,
			out_features=d_input,
			use_bias=use_bias,
			**shared_kwargs,
		)

	def __call__(self, x, attention_mask=None, segment_ids=None, bias=None):
		residual = x
		norm_x = self.norm(x)

		query = self.w_query(norm_x)
		key = self.w_key(norm_x)
		value = self.w_value(norm_x)

		query = query.reshape(query.shape[0], query.shape[1], self.num_heads, self.head_size)
		key = key.reshape(key.shape[0], key.shape[1], self.num_heads, self.head_size)
		value = value.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_size)
		
		if (bias is None) and (attention_mask is not None) and (segment_ids is not None):
			bias = self.attn_bias_fn(attention_mask, segment_ids, fully_causal=self.fully_causal)

		attention_output = self.attn_dpa_fn(
			query=query,
			key=key,
			value=value,
			bias=bias,
		)

		attention_output = attention_output.reshape(attention_output.shape[0], attention_output.shape[1], -1)

		output = self.w_output(attention_output)
		return residual + output
