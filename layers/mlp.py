
import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
import utils
from typing import Optional


class MLP(nnx.Module):
	'''
	A simple multi-layer perceptron.
	'''

	def __init__(
			self,
			d_input: int,
			d_mlp: int,
			use_bias=False,
			param_dtype='float32',
			compute_dtype='float32',
			activation='sigmoid',
			rngs=None,
			**kwargs,
		):

		param_dtype = utils.str_to_jax_dtype(param_dtype)
		compute_dtype = utils.str_to_jax_dtype(compute_dtype)
		self.activation_fn = utils.str_to_jax_activation(activation)

		shared_kwargs = dict(
			param_dtype=param_dtype,
			dtype=compute_dtype,
			rngs=rngs,
		)

		self.norm = nnx.RMSNorm(
			num_features=d_input,
			**shared_kwargs,
		)

		self.w_input_0 = nnx.Linear(
			in_features=d_input,
			out_features=d_mlp,
			use_bias=use_bias,
			**shared_kwargs,
		)

		self.w_input_1 = nnx.Linear(
			in_features=d_input,
			out_features=d_mlp,
			use_bias=use_bias,
			**shared_kwargs,
		)

		self.w_output = nnx.Linear(
			in_features=d_mlp,
			out_features=d_input,
			use_bias=use_bias,
			**shared_kwargs,
		)


	def __call__(self, x):
		residual = x
		norm_x = self.norm(x)
		wi0 = self.w_input_0(norm_x)
		wi1 = self.w_input_1(norm_x)
		intermediate = (self.activation_fn(wi0) * wi1)
		w_out = self.w_output(intermediate)
		return (residual + w_out)