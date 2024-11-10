# Minimal model using Flax's new NNX api

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.training.common_utils import onehot


   
def log_softmax_cross_entropy(logits, label_ids, loss_mask=None, use_token_length_normalization=False, loss_dtype=jnp.float32):
	'''
	Uses log_softmax cross-entropy loss for stability.
	Supports loss_masking by setting loss_mask to 0 for tokens to ignore. Set loss_mask to 1 to disable masking.
	Returns per-sample losses.
	
	Input shapes and types:
		logits: (B, L, D), float32 (or will be casted)
		label_ids.shape: (B, L), int
		loss_mask.shape: (B, L), float32
		use_token_length_normalization: bool
			- Not a feature in the EleutherAI harness. Used in Birdie.

	Output shapes and types:
		loss: (B,), float32

   '''

	logits = logits.astype(loss_dtype)
	logits_max = jnp.max(logits, axis=-1, keepdims=True)
	shifted = (logits - jax.lax.stop_gradient(logits_max))
	shifted_logsumexp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))
	log_softmax = (shifted - shifted_logsumexp)
	

	labels = onehot(label_ids, logits.shape[-1])
	logits, labels = logits.astype(loss_dtype), labels.astype(loss_dtype)


	# (B, L,)
	loss = -jnp.sum(labels * log_softmax, axis=-1) 

	if loss_mask is not None:
		loss *= loss_mask

		if use_token_length_normalization:
			# Normalizes the loss by the number of tokens
			loss /= jnp.maximum(jnp.sum(loss_mask, axis=-1, keepdims=True,), 1.0)  

	return loss

def perplexity(logits, label_ids, loss_mask=None, loss_dtype=jnp.float32):
	'''
	Calculates perplexity from logits (B, L, D) and label_ids (B, L).
	'''
	loss = log_softmax_cross_entropy(logits, label_ids, loss_mask=loss_mask, loss_dtype=loss_dtype)
	if loss_mask is not None:
		mean_loss = jnp.mean(loss, axis=-1, where=(loss_mask == 1))  # (B, L) -> (B,)
	else:
		mean_loss = jnp.mean(loss, axis=-1)
	perplexity = jnp.exp(mean_loss)
	return perplexity


class Model(nnx.Module):
	def __init__(self, din, dmid, dout, rngs: nnx.Rngs):
		self.linear = nnx.Linear(din, dmid, rngs=rngs)
		self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

def __call__(self, x):
	x = nnx.relu(self.dropout(self.bn(self.linear(x))))
	return self.linear_out(x)


@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer, x, y):
	def loss_fn(model):
		y_pred = model(x)  # call methods directly
		return ((y_pred - y) ** 2).mean()

	loss, grads = nnx.value_and_grad(loss_fn)(model)
	optimizer.update(grads)  # in-place updates

	return loss


def get_model(**kwargs):
	'''
	This is called from host_main.py
	kwargs can contain whatever you need. Thanks Python!
	'''

	model = Model(2, 64, 3, rngs=nnx.Rngs(0))  # eager initialization
	# optimizer = nnx.Optimizer(model, optax.adam(1e-3))  # reference sharing	
