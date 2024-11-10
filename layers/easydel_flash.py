usage = '''


clear; python /scratch/sblouir/mount/code/f22/new_flash.py;

./dt_interactive.sh /scratch/sblouir/mount/code/f22/new_flash.py


pip install git+https://github.com/erfanzar/jax-flash-attn2
pip install git+https://github.com/deepmind/chex.git
pip install einops
pip install triton

'''

if __name__ == "__main__":
	import xla_cfg
	import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial



# attention_bias = attention_bias[..., None, None, :]


def caster(x):
	return jnp.float16(x)



# @partial(jax.jit, static_argnums=2)	
def calculate_bias(attention_mask, segment_ids, fully_causal=True, dtype=jnp.float16,):
	if fully_causal:
		mask = nn.make_causal_mask(attention_mask)
	else:
		mask = jnp.logical_or(nn.make_attention_mask(attention_mask, attention_mask, ), nn.make_causal_mask(attention_mask))
	segment_mask = nn.make_attention_mask(segment_ids, segment_ids, pairwise_fn=jnp.equal, )
	mask = nn.combine_masks(mask, segment_mask)
	bias = jax.lax.select(
		mask > 0, 
		jnp.full(mask.shape, 0., dtype=dtype,),
		jnp.full(mask.shape, -1e6, dtype=dtype,),
	)
	return bias



# Get a cached attention instance
# attention = get_cached_flash_attention(
# 	backend="gpu", # 'gpu', 'tpu', or 'cpu'
# 	# platform="pallas", # 'triton', 'pallas', or 'jax'
# 	platform="triton", # 'triton', 'pallas', or 'jax'
# 	blocksize_q=128, # BLOCK SIZE Q
# 	blocksize_k=128, # BLOCK SIZE K
# 	softmax_scale=(128 ** -0.5) # Optional scaling factor
# 	# softmax_scale=(query.shape[-1] ** -0.5) # Optional scaling factor
# )

from jax_flash_attn2 import get_cached_flash_attention
# @jax.jit
def run_mhdpa(query, key, value, bias=None, attention_mask=None, segment_ids=None,):

	print(f'  new_flash.py:  Using new_flash.run_mhdpa()!')

	query = caster(query)
	key = caster(key)
	value = caster(value)

	# if (bias is None) and (attention_mask is not None) and (segment_ids is not None):
	# bias = calculate_bias(attention_mask=attention_mask, segment_ids=segment_ids, fully_causal=False,)

	if bias is None:
		bias = jnp.zeros_like(query[..., :1, :1, :])
	bias = caster(bias)
	print(f"  bias.shape: {bias.shape}")


	assert(len(query.shape) == 4), f"query.shape: {query.shape}"

	# Use with your tensors
	# attention = jax.v
	# query = query[:, None]
	# key = key[:, None]
	# value = value[:, None]
	# bias = bias[:, None]
	# outputs = jax.vmap(attention)(
	# # outputs = attention(
	# 	query=query,
	# 	key=key,
	# 	value=value,
	# 	bias=bias,
	# # )


	# # Get a cached attention instance
	attention = get_cached_flash_attention(
		backend="gpu", # 'gpu', 'tpu', or 'cpu'
		# platform="jax", # 'triton', 'pallas', or 'jax'
		platform="triton", # 'triton', 'pallas', or 'jax'
		blocksize_q=64, # BLOCK SIZE Q
		blocksize_k=64, # BLOCK SIZE K
		# softmax_scale=(128 ** -0.5) # Optional scaling factor
		softmax_scale=(query.shape[-1] ** -0.5) # Optional scaling factor
	)

	def chunked_attention(query, key, value, bias, chunk_size):
		def process_chunk(state, inputs):
			q_chunk, k_chunk, v_chunk, b_chunk = inputs

			print(f"*" * 60,)
			print(f"  q_chunk.shape: {q_chunk.shape}")
			print(f"  k_chunk.shape: {k_chunk.shape}")
			print(f"  v_chunk.shape: {v_chunk.shape}")
			print(f"  b_chunk.shape: {b_chunk.shape}")
			print(f"*" * 60,)
			output = (attention)(query=q_chunk, key=k_chunk, value=v_chunk, bias=b_chunk)
			return state, output

		# Reshape inputs to have shape (num_chunks, chunk_size, ...)
		total_length = query.shape[0]
		# num_chunks = (total_length + chunk_size - 1) // chunk_size  # Calculate number of chunks, rounding up
		num_chunks = total_length // chunk_size
		num_chunks = max(1, num_chunks)
		chunk_size = min(chunk_size, total_length//num_chunks)

		# query = jax.lax.pad(query, padding_value=0, padding_config=((0, num_chunks * chunk_size - total_length, 0),) + ((0, 0),) * (query.ndim - 1))
		# key = jax.lax.pad(key, padding_value=0, padding_config=((0, num_chunks * chunk_size - total_length, 0),) + ((0, 0),) * (key.ndim - 1))
		# value = jax.lax.pad(value, padding_value=0, padding_config=((0, num_chunks * chunk_size - total_length, 0),) + ((0, 0),) * (value.ndim - 1))
		# bias = jax.lax.pad(bias, padding_value=1e-6, padding_config=((0, num_chunks * chunk_size - total_length, 0),) + ((0, 0),) * (bias.ndim - 1))

		query_chunks = query.reshape(num_chunks, chunk_size, *query.shape[1:])
		key_chunks = key.reshape(num_chunks, chunk_size, *key.shape[1:])
		value_chunks = value.reshape(num_chunks, chunk_size, *value.shape[1:])
		bias_chunks = bias.reshape(num_chunks, chunk_size, *bias.shape[1:])
		# query_chunks = query[None]
		# key_chunks = key[None]
		# value_chunks = value[None]
		# bias_chunks = bias[None]

		print(f"  query_chunks.shape: {query_chunks.shape}")
		print(f"  key_chunks.shape: {key_chunks.shape}")
		print(f"  value_chunks.shape: {value_chunks.shape}")
		print(f"  bias_chunks.shape: {bias_chunks.shape}")

		# Use scan to process each chunk
		_, outputs = jax.lax.scan(process_chunk, None, (query_chunks, key_chunks, value_chunks, bias_chunks))
		return outputs.reshape(total_length, *outputs.shape[2:])
	
	# outputs = chunked_attention(query, key, value, bias, chunk_size=2)
	outputs = attention(query=query, key=key, value=value, bias=bias)

	print(f"  outputs.shape: {outputs.shape}")



	return outputs

	# outputs = outputs.reshape(query.shape[0], query.shape[1], -1)

if __name__ == "__main__":

	seeded_rng = np.random.default_rng(0)

	batch = 8
	length = 128
	headdim = 64
	dims = 256

	batch = 8
	length = 16384
	headdim = 128
	dims = 2048

	batch = 4
	length = 16384
	headdim = 128
	dims = 2048

	num_heads = (dims // headdim)

	query_states = seeded_rng.random((batch, length, num_heads, headdim))
	key_states = seeded_rng.random((batch, length, num_heads, headdim))
	value_states = seeded_rng.random((batch, length, num_heads, headdim))
	attention_mask = seeded_rng.random((batch, length)).astype(np.int32)



	segment_ids = []

	for b in range(batch):
		segment_id = 1
		current_segment = []
		for s in range(length):
			current_segment.append(segment_id)
			random_number = seeded_rng.random()
			if random_number < 0.08:
				segment_id += 1
		segment_ids.append(current_segment)
	segment_ids = jnp.int32(segment_ids)
		

	print(f"*" * 60,)
	print(f"  query_states.shape: {query_states.shape}")
	print(f"  key_states.shape: {key_states.shape}")
	print(f"  value_states.shape: {value_states.shape}")
	print(f"  attention_mask.shape: {attention_mask.shape}")
	print(f"  segment_ids.shape: {segment_ids.shape}")

	query_states = caster(query_states)
	key_states = caster(key_states)
	value_states = caster(value_states)
	attention_mask = caster(attention_mask)
	segment_ids = caster(segment_ids)

	# outputs = query_states + key_states + value_states

	bias = calculate_bias(attention_mask=attention_mask, segment_ids=segment_ids, fully_causal=False,)


	outputs = run_mhdpa(query_states, key_states, value_states, bias=bias,)
	# outputs = run_mhdpa(query_states, key_states, value_states, attention_mask=attention_mask, segment_ids=segment_ids,)

	print(f"  outputs.shape: {outputs.shape}")
	print(f"*" * 60,)



