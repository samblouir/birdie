
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import os
import json
import pickle

def str_to_jax_activation(x):
	assert(isinstance(x, str)), f"Expected a string, got {type(x)}"

	try:
		if x in ['gelu',]:
				return nnx.gelu
		elif x in ['sigmoid',]:
			return nnx.sigmoid
		elif x in ['tanh',]:
			return nnx.tanh
		elif x in ['relu',]:
			return nnx.relu
		elif x in ['elu',]:
			return nnx.elu
		elif x in ['leaky_relu',]:
			return nnx.leaky_relu
		elif x in ['selu',]:
			return nnx.selu
	except ValueError:
		raise ValueError(f'Unknown activation argument: {x}')
	


def str_to_jax_dtype(x):
	if isinstance(x, jnp.dtype):
		return x
	# assert(isinstance(x, str)), f"Expected a string, got {type(x)}"
	
	try:
		if x in ['fp32', 'float32']:
			return jnp.float32
		elif x in ['bf16', 'bfloat16']:
			return jnp.bfloat16
		elif x in ['fp16', 'float16']:
			return jnp.float16
		elif x in ['int32', 'int']:
			return jnp.int32
		elif x in ['int16',]:
			return jnp.int16
		elif x in ['int8', ]:
			return jnp.int8
		elif x in ['uint8', ]:
			return jnp.uint8
		elif x in ['c64', 'complex64', 'complex', ]:
			return jnp.complex64
		elif x in ['c128', 'complex128', 'complex', ]:
			return jnp.complex128
		else:
			return x
	except ValueError:
		raise ValueError(f'Unknown dtype argument: {x}')
	



def make_divisible_by(x, divisor, round_up=True,):
	diff = (x % divisor)
	if round_up:
		return (x + diff)
	else:
		return (x - diff)

def make_power_of_2(x, round_up=True,):
	if round_up:
		ret_val = (x - 1)
	else:
		ret_val = (x)
	return int(2 ** ret_val.bit_length())




def load_cfg(path):

	json_path = f"{path.rsplit('.',1)[0]}.json"

	if not os.path.exists(json_path):
		# assert(os.path.exists(path)), f"Config file not found at \"{path}\""

		if not os.path.exists(path):
			name_txt = path.rsplit('/', 1)[0] + '/name.txt'
			print(f"  name_txt: {name_txt}")
			with open(name_txt, 'r') as f:
				name = f.read().strip()

			names = [
				f"{name}",
				f"{name}^mls:16384__trg:32__tns:1000__acm:32__ns:250__lr1.0e-04__eps1e-15__fes:6",
				f"{name}^load_save^flanCollection25^load_save^h2",
				f"{name}^mls:16384__trg:8__tns:800__acm:64__ns:100__lr5.0e-05__eps1e-15__fes:6__d:9__flr:1",
			]

			print(f"*" * 60,)
			for name in names:
				source_model_arg_dict_path = os.path.join("/scratch/sblouir/testMC4vocab-Logs/", name, 'model_arg_dict.pkl')
				print(f"  name: {name}")
				print(f"  source_model_arg_dict_path: {source_model_arg_dict_path}")
				if not os.path.exists(source_model_arg_dict_path):
					continue
					
				target_model_arg_dict_path = path

				os.system(f"cp {source_model_arg_dict_path} {target_model_arg_dict_path}")
				print(f"  Copied model_arg_dict.pkl from \"{source_model_arg_dict_path}\" to \"{target_model_arg_dict_path}\"")
				break


		with open(path, 'rb') as f:
			cfg = pickle.load(f)

		# save as json
		with open(json_path, 'w') as f:
			json.dump(cfg, f, indent=4)
		print(f"  Saved config to \"{json_path}\"")
	

	with open(json_path, 'r') as f:
		cfg = json.load(f)
		print(f"  Loaded config from \"{json_path}\"")

	return cfg



if __name__ == "__main__":

	models = [
		# "attn_mini",
		# "attn_clm",
		# "attn_birdie",
		# "gated_ssm_birdie",
		# "gated_ssm_frm",
		# "hawk_birdie",
		# "hawk_birdie_causal",
		# "hawk_clm",

		
		"attn_birdie",
		"attn_clm",
		"attn_mini",
		"gated_ssm_birdie_causal",
		"gated_ssm_birdie",
		"gated_ssm_clm",
		"gated_ssm_frm",
		"gated_ssm_ul2",
		"hawk_birdie_causal",
		"hawk_birdie",
		"hawk_clm",
	]

	full_pkl_paths = [f"saved_models/{model}/model_arg_dict.pkl" for model in models]
	

	seen_opts = {}
	for pkl_path in full_pkl_paths:
		print(f"pkl_path: {pkl_path}")
		cfg = load_cfg(pkl_path)

		for cfg_idx, (key, value) in enumerate(cfg.items()):
			seen_opts[key] = seen_opts.get(key, 0) + 1

	print("keys_to_keep = {")
	for seen_opts_idx, (key, value) in enumerate(seen_opts.items()):
		if value < 9:
			print(f"	\"{key}\": {value},")
	print("}")

	keys_to_keep = {
		"fully_causal": 2,
		"head_dims": 3,
		"num_heads": 3,
		"pretraining_objective": 2,
		"striped_encoder_decoder": 8,
		"use_hawk": 3,
		"vocab_size": 1,
	}



		
		