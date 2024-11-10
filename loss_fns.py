import jax
import jax.numpy as jnp
from flax.training.common_utils import onehot


''' 
	This provides functions for calculating loss and perplexity, using a loss_mask to ignore padding tokens.
	it uses log_softmax_cross_entropy to improve numerical stability.
	This can be changed to just checking for label_ids == 0, or label_ids == -100 (as in HuggingFace's implementations).
'''

   
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
		loss: (B, L), float32

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
