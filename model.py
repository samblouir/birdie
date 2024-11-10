# Minimal model using Flax's new NNX api


import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training.common_utils import onehot
import functools
from typing import Optional

from layers import attention
from layers import mlp
from layers import embeddings

import utils
import loss_fns


@nnx.jit
def train_step(model, optimizer, input_ids, label_ids, loss_mask=None, attention_mask=None, segment_ids=None, reset_mask=None,):
	
	def loss_fn(model):
		logits = model(
			input_ids=input_ids,
			segment_ids=segment_ids,
			attention_mask=attention_mask,
			reset_mask=reset_mask,
		)
		loss = loss_fns.log_softmax_cross_entropy(logits=logits, label_ids=label_ids, loss_mask=loss_mask)
		return jnp.mean(loss)

	loss, grads = nnx.value_and_grad(loss_fn)(model)
	optimizer.update(grads)

	return loss


class LP(nnx.Module):
	def __init__(self, d_input, d_out, rngs, **kwargs):
		self.d_input = d_input
		self.d_out = d_out
		self.rngs = rngs

		self.linear = nnx.Linear(
			in_features=d_input,
			out_features=d_out,
			rngs=rngs,
		)

	def __call__(self, x):
		x = self.linear(x)
		return x
	

class TransformerBlock(nnx.Module):
	def __init__(self,
			d_input: Optional[int],
			d_mlp: Optional[int],
			rngs: Optional[int],
			fully_causal: Optional[bool] = False,
			head_size: Optional[int] = None,
			use_bias: Optional[bool] = False,
			attn_fn: Optional[str] = None,
			param_dtype: Optional[str] = 'float32',
			compute_dtype: Optional[str] = 'float32',
			**kwargs,
			):

		self.attention = attention.MultiheadedAttention(
			d_input=d_input,
			fully_causal=fully_causal,
			head_size=head_size,
			use_bias=use_bias,
			attn_fn=attn_fn,
			param_dtype=param_dtype,
			compute_dtype=compute_dtype,
			rngs=rngs,
		)

		self.mlp = mlp.MLP(
			d_input=d_input,
			d_mlp=d_mlp,
			param_dtype=param_dtype,
			compute_dtype=compute_dtype,
			rngs=rngs,
		)

	def __call__(self, x):
		x = self.attention(x)
		x = self.mlp(x)
		return x

class Model(nnx.Module):
	def __init__(self,
			vocab_size: Optional[int],
			d_input: Optional[int],
			d_mlp: Optional[int],
			rngs: Optional[int],
			fully_causal: Optional[bool] = False,
			head_size: Optional[int] = None,
			use_bias: Optional[bool] = False,
			param_dtype: Optional[str] = 'float32',
			compute_dtype: Optional[str] = 'float32',
			attn_fn: Optional[str] = None,
			tied_embeddings: Optional[bool] = False,
			num_layers: Optional[int] = 12,
			**kwargs,
		):


		param_dtype = utils.str_to_jax_dtype(param_dtype)
		compute_dtype = utils.str_to_jax_dtype(compute_dtype)

		shared_kwargs = dict(
			rngs=rngs,
			param_dtype=param_dtype,
			compute_dtype=compute_dtype,
		)


		self.embeddings = embeddings.Embeddings(
			d_embeddings=d_input,
			vocab_size=vocab_size,
			**shared_kwargs,
		)

		if tied_embeddings:
			def attend_fn(x):
				x = self.embeddings.attend(x)
				return x / jnp.sqrt(d_input) # Scaled to keep input embeddings happy
			self.vocab_head_proj = attend_fn
		else:
			self.vocab_head_proj = nnx.Linear(
				in_features=d_input,
				out_features=vocab_size,
				rngs=rngs,
			)

		self.vocab_head_norm = nnx.RMSNorm(
			num_features=d_input,
			param_dtype=param_dtype,
			rngs=rngs,
		)

		def vocab_head(x):
			x = self.vocab_head_norm(x)
			x = self.vocab_head_proj(x)
			return x
		
		self.vocab_head = vocab_head

		
		@nnx.split_rngs(splits=num_layers)
		@nnx.vmap(in_axes=(0,), out_axes=0)
		def create_block(rngs: nnx.Rngs):
			return TransformerBlock(
				d_input=d_input,
				d_mlp=d_mlp,
				fully_causal=fully_causal,
				head_size=head_size,
				use_bias=use_bias,
				attn_fn=attn_fn,
				rngs=rngs,
				param_dtype=param_dtype,
				compute_dtype=compute_dtype,
			)
		
		self.blocks = create_block(rngs)
		self.num_layers = num_layers

		@nnx.split_rngs(splits=num_layers)
		@nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
		def block_forward(x, layer):
			x = layer(x)
			return x
		
		self.block_forward = block_forward
			


	@nnx.jit
	def __call__(self, input_ids):
		x = self.embeddings(input_ids)
		x = self.block_forward(x, self.blocks)
		x = self.vocab_head(x)
		return x




def get_model(**kwargs):
	'''
	This is called from host_main.py
	kwargs can contain whatever you need. Thanks Python!
	'''

	model_kwargs = {
		**kwargs,
		**dict(
			vocab_size=kwargs.get('vocab_size', 32),
			d_input=kwargs.get('d_input', 2),
			d_hidden=kwargs.get('d_hidden', 64),
			d_out=kwargs.get('d_out', 3),
			rngs=nnx.Rngs(kwargs.get('model_param_rng_seed', 0)),
		),
	}
	model = Model(**model_kwargs,)
	# optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing
	return model





