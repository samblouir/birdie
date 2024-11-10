import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx
import utils
from typing import Optional

class Embeddings(nnx.Module):
	'''
	Embeddings layer that maps input_ids to dense vector representations.
	'''

	def __init__(
			self,
			d_embeddings: int,
			vocab_size: int,
			rngs: int,
			param_dtype: str = 'float32',
			embedding_out_dtype: str = 'float32',
			**kwargs,
		):
		self.d_embeddings = d_embeddings
		self.vocab_size = vocab_size
		self.rngs = rngs
		self.param_dtype = param_dtype
		self.embedding_out_dtype = embedding_out_dtype

		# Additional keyword arguments are captured in `kwargs`.
		for key, value in kwargs.items():
			setattr(self, key, value)

		param_dtype = utils.str_to_jax_dtype(self.param_dtype)
		embedding_out_dtype = utils.str_to_jax_dtype(self.embedding_out_dtype)

		self.embeddings = nnx.Embed(
			num_embeddings=self.vocab_size,
			features=self.d_embeddings,
			param_dtype=param_dtype,
			dtype=embedding_out_dtype,
			rngs=self.rngs,
		)

	def __call__(self, input_ids):
		'''
		Maps input_ids to their corresponding embeddings.

		Args:
			input_ids: A tensor of shape (B, L) containing integer indices.

		Returns:
			A tensor of shape (B, L, D) containing the embeddings for the input_ids.
		'''
		embeddings = self.embeddings(input_ids)
		return embeddings

	def attend(self, x):
		'''
		A placeholder method for attending to the input embeddings.

		Args:
			x: A tensor of shape (B, L, D) representing embedded inputs.

		Returns:
			A tensor of shape (B, L, vocab_size) resulting from attending to the embeddings.
		'''
		return self.embeddings.attend(x)
