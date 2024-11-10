'''

	This is the original span corruption and auto-encoding implementation as used in the paper.

	It has NOT been refactored yet.
	You have been warned!

'''

import jax
import ctypes
import time
import os
import numpy as np
import ctypes
import hashlib
import pickle
import os
from dataclasses import dataclass, field
is_tpu = (os.environ.get("py_version", "") == "3.10")

independent_storage_across_processes = (os.environ.get("independent_storage_across_processes", "") == "0")



@dataclass
class SpanCorruptionConfig:
	mean_span_width: float = 0
	mean_corruption_percentage: float = 0.0
	use_autoencoding_objective: bool = False
	max_length_mult: float = 4
	sentinel_start: int = 31900
	sentinel_end: int = 32000
	min_length: int = 16
	max_length:int = 1_000_000
	prefix_text: str = ""
	minimum_corruption_percentage: float = field(default=None)
	maximum_corruption_percentage: float = field(default=None)
	fitm: int = 0
	deshuffle: int = 0
	prompt_pfx:int = 0
	objective:str = ""
	num_queries:int = 2
	generator_fn: callable = None
	generator_fn_kwargs: dict = field(default_factory=dict)
	built_generator_fn: callable = None
	generator_iterator: callable = None

	min_length_mult: float = 1.0

	def build_generator_fn(self, kwargs=None, seed_offset=0,):
		if kwargs is None:
			kwargs = {}
		final_kwargs = {**self.generator_fn_kwargs, **kwargs}
		final_kwargs['np_rng_seed'] = final_kwargs.get('np_rng_seed', 0) + seed_offset
		self.generator_iterator = self.generator_fn(**final_kwargs)
		self.built_generator_fn = lambda *args, **kwargs: next(self.generator_iterator)

	def __post_init__(self, ):

		self.minimum_corruption_percentage = np.floor(self.mean_corruption_percentage * 0.3)
		self.maximum_corruption_percentage = np.ceil(self.mean_corruption_percentage * 2.0)

		self.maximum_corruption_percentage = min(1.0, self.maximum_corruption_percentage)

		msw = self.mean_span_width
		mcp = self.mean_corruption_percentage

		try:
			len_frac = max(msw+1, int( ((msw / mcp) + msw)/2 ))
		except Exception as e:
			len_frac = 1
		
		if mcp == 0:
			mcp = 1
		len_frac = max(msw+1, int( ((msw / mcp) + msw)/3 ))
  
		if self.objective in ['span_corruption', 'ssmPT', ]:
			self.min_length = max(self.min_length, len_frac + 16)


		msgs = [
			f"self.minimal_corruption_percentage: {float(self.minimum_corruption_percentage):0.2f}",
			f"self.maximum_corruption_percentage: {float(self.maximum_corruption_percentage):0.2f}",
			f"self.mean_span_width: {float(self.mean_span_width):0.2f}",
			f"self.mean_corruption_percentage: {float(self.mean_corruption_percentage):0.2f}",
			f"self.min_length: {int(self.min_length):,}",
		]
	
	def __str__(self,):
		current_str = ""
		for k, v in self.__dict__.items():
			current_str += f"{k}: {v}\n"
		if USER == "sblouir":
			return f"{self.prefix_text}SpanCorruptionConfig {current_str}"
		return ""
	
	def get_prefix_text(self):
		return f"{self.prefix_text}"
	
	# called when ['key'] is used
	def __getitem__(self, key):
		return getattr(self, key)
	
	# called when ['key'] is used
	def __setitem__(self, key, value):
		setattr(self, key, value)
	
	# accept keys and set values
	def set_values(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		return self
	
	def set_objective(self, objective):
		self.objective = objective
		return self
	



@dataclass
class SpanCorruptionInstance:
	inputs: np.ndarray
	max_allowed_length: int
	rng_seed: int = 0
	max_spans: int = 0
	corruption_config: SpanCorruptionConfig = None

	def __getattr__(self, name):
		# Falls back to corruption_config
		return getattr(self.corruption_config, name)
	
	def __str__(self,):
		current_str = ""
		for k, v in self.__dict__.items():
			if k == "corruption_config":
				continue
			current_str += f"\n\t{k}: {v}"
		return f"{self.prefix_text}SpanCorruptionConfig {current_str}"
	




















def get_hash_of_object(in_obj):
	in_obj = f"{in_obj}"

	in_obj_dumps = pickle.dumps(in_obj)

	ret_val = hashlib.sha256(in_obj_dumps).hexdigest()
	return f"{ret_val}"

def get_hash_of_file_at_file_path(in_path):
	with open(in_path, 'r') as f:
		text = f.read()
	return get_hash_of_object(text)

def maybe_compile_cpp(cpp_path, pid=0,):

	cpp_file_hash = get_hash_of_file_at_file_path(cpp_path)
	file_base_dir = f"{cpp_path}".rsplit("/", 1)[0]

	so_path = f"{file_base_dir}/tmp/{cpp_file_hash}.so"

	if not independent_storage_across_processes:
		while (pid > 0) and (not os.path.exists(so_path)):
			print(f"  maybe_compile_cpp(...):    process #({pid}) is waiting for {so_path} to be compiled...")
			time.sleep(1)

	if not os.path.exists(so_path):
		os.system(f"mkdir -p {file_base_dir}/tmp")
		cmd = []
		cmd.append(f"cd {file_base_dir}")
		cmd.append(f"ls -al")
		cmd.append(f"ml gnu10")
		cmd.append(f"g++ -shared -o {so_path} -fPIC {cpp_path}")
		cmd = '; '.join(cmd)
		print(f"  Compiling C++ code! Storing .so at \"{so_path}\"...")
		print(f"    cmd:  \"{cmd}\"")
		os.system(cmd)
		print(f"  Finished compiling C++!")
	return so_path


def load_cpp(cpp_path, pid=0,):
	so_path = maybe_compile_cpp(cpp_path, pid=pid,)
	lib = ctypes.CDLL(so_path)
	return lib


def prep_lib():
	file_loc = f"{__file__}".rsplit("/", 1)[0]
	cpp_path = f"{file_loc}/span.cpp"
	lib = load_cpp(cpp_path, pid=jax.process_index(),)
	lib.should_we_corrupt_this_span.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_int]
	lib.run_span_corruption.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ]
	lib.run_ssm_span_corruption.argtypes = [np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), np.ctypeslib.ndpointer(dtype=np.int32), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ]
	return lib

lib = prep_lib()


def corrupt_span_iterative(input_array, max_allowed_length=100, mean_corrupt_chance=0.15, mean_span_width=3, sentinel_start=31900, sentinel_end=32000,  rng_seed=-1, *args, **kwargs,):

	labels = []
	while len(labels) < 1:
		ready = lib.get_is_ready()
		while not ready:
			# lib.set_not_ready()
			ready = lib.get_is_ready()
			# print(f"  ready: {ready}", file=sys.stderr)

		rv = run_span_corruption(input_array, max_allowed_length, mean_corrupt_chance, mean_span_width, sentinel_start, sentinel_end, rng_seed, **kwargs,)
		if rv is not None:
			(new_x, corrupted_input, labels, num_not_corrupted, num_corrupted,) = rv

		ready = lib.get_is_ready()
		while not ready:
			ready = lib.get_is_ready()

	corrupted_input = np.stack(corrupted_input)
	labels = np.stack(labels)

	actual_corruption_rate = np.float32([(num_corrupted / (num_corrupted + num_not_corrupted))])
	rd = dict(inputs=corrupted_input, labels=labels, new_x=new_x, actual_corruption_rate=actual_corruption_rate,)
	return rd




def run_span_corruption(input_array:np.ndarray, max_allowed_length:int, mean_corrupt_chance:float, mean_span_width:int, sentinel_start:int, sentinel_end:int, rng_seed=0, np_rng:np.random.Generator=None, use_autoencoding_objective=False, fitm=0, **kwargs,):
	input_array = np.int32(input_array)
	initial_cutoff = max(1, (max_allowed_length - 2))
	originally_sliced = input_array[initial_cutoff:]
	input_array = input_array[:initial_cutoff]

	og_len = input_array.shape[0]
	results_array = np.zeros(4, dtype=np.int32)
	if use_autoencoding_objective:
		fn = lib.run_ssm_span_corruption
		# label_offset = 1
	else:
		fn = lib.run_span_corruption
		# label_offset = 0
	
	loop_counter = 0
	while results_array[1] == 0:
		out_corrupted_input = np.zeros((og_len+8,), dtype=input_array.dtype)
		out_labels = np.zeros((og_len+8,), dtype=input_array.dtype)
		results_array = np.zeros(4, dtype=np.int32)
		out_length = fn(input_array, out_corrupted_input, out_labels, results_array, len(out_corrupted_input), len(out_labels), og_len, max_allowed_length, mean_corrupt_chance, mean_span_width, sentinel_start, sentinel_end)
		loop_counter += 1
		if (loop_counter % 100 == 0):

			# print(f"  testSimpleSpan.py  run_span_corruption(...):  Warning! loop_counter: {loop_counter}", flush=True,)
			msgs = []
			msgs.append(f"\n" * 3, )
			msgs.append(f"*" * 60,)
			msgs.append(f"  testSimpleSpan.py  run_span_corruption(...):  Warning! loop_counter: {loop_counter:,}")
			msgs.append(f"\n  input_array {input_array.shape}: \n{input_array}\n")
			msgs.append(f"\n  results_array {results_array.shape}: \n{results_array}\n")
			msgs.append(f"\n  out_corrupted_input {out_corrupted_input.shape}: \n{out_corrupted_input}\n")
			msgs.append(f"\n  out_labels {out_labels.shape}: \n{out_labels}\n")
			msgs.append(f"\n  out_length: {out_length}")
			msgs.append(f"\n max_allowed_length: {max_allowed_length}")
			msgs.append(f"\n mean_corrupt_chance: {mean_corrupt_chance}")
			msgs.append(f"\n mean_span_width: {mean_span_width}")
			msgs.append(f"\n sentinel_start: {sentinel_start}")
			msgs.append(f"\n sentinel_end: {sentinel_end}")
			msgs.append(f"\n rng_seed: {rng_seed}")
			msgs.append(f"\n use_autoencoding_objective: {use_autoencoding_objective}")
			msgs.append(f"\n  initial_cutoff: {initial_cutoff}")
			msgs.append(f"\n  og_len: {og_len}")
			msgs.append(f"*" * 60,)
			print('\n'.join(msgs), flush=True,)
			return None
			# exit()

	
	l0 = results_array[0]
	l1 = results_array[1]
	num_not_corrupted = results_array[2]
	num_corrupted = results_array[3]
	
	# l0 = max(l0, max_allowed_length)
	# l1 = max(l1, max_allowed_length)
	
	# _out_corrupted_input = np.zeros(l0, dtype=np.int32)
	# _out_labels = np.zeros(l1, dtype=np.int32)
	# _out_corrupted_input += out_corrupted_input[:len(_out_corrupted_input)]
	# _out_labels += out_labels[:len(_out_labels)]
	# out_corrupted_input = _out_corrupted_input
	# out_labels = _out_labels
	out_corrupted_input = out_corrupted_input[:l0]
	out_labels = out_labels[:l1]
	
	input_array = input_array[(num_corrupted + num_not_corrupted):]
	input_array = np.concatenate([input_array, originally_sliced])
	
	
	del results_array
	
	
	return input_array, out_corrupted_input, out_labels, num_not_corrupted, num_corrupted

































	


def find_valid_result(corruption_instance, use_autoencoding_objective=False, db=0, rng_ctr=-1, corruption_config=None, remaining_space=-1, gl=-1,):

	if corruption_config is None:
		corruption_config = corruption_instance.corruption_config
	rng_ctr = corruption_instance.rng_seed


	inputs = corruption_instance.inputs
	mean_span_width = corruption_instance.mean_span_width
	mean_corruption_percentage = corruption_instance.mean_corruption_percentage
	minimum_corruption_percentage = corruption_instance.minimum_corruption_percentage
	maximum_corruption_percentage = corruption_instance.maximum_corruption_percentage
	sentinel_start = corruption_instance.sentinel_start
	sentinel_end = corruption_instance.sentinel_end
	max_length_mult = corruption_instance.max_length_mult
	min_length = corruption_instance.min_length
	max_length = corruption_instance.max_length
	rng_seed = corruption_instance.rng_seed
	max_allowed_length = corruption_instance.max_allowed_length
	use_autoencoding_objective = corruption_instance.use_autoencoding_objective

	use_autoencoding_objective = corruption_config.use_autoencoding_objective
	fitm = corruption_config.fitm
	
	rng_seed = np.abs(rng_seed)
	np_rng = np.random.default_rng(rng_seed)

	current_corruption_percentage = np_rng.normal(mean_corruption_percentage)
	current_corruption_percentage = max(minimum_corruption_percentage, current_corruption_percentage)
	current_corruption_percentage = min(maximum_corruption_percentage, current_corruption_percentage)

	span_corruption_kwargs = dict(
		mean_span_width=mean_span_width,
		mean_corruption_percentage=mean_corruption_percentage,
		sentinel_start=sentinel_start,
		sentinel_end=sentinel_end,
		max_length_mult=max_length_mult,
		rng_seed=rng_seed,
		fitm=fitm,
	)

	pp_kwargs = dict(
		original_length=len(inputs),
		fitm=fitm,
		use_autoencoding_objective=use_autoencoding_objective,
		current_corruption_percentage=current_corruption_percentage,
		np_rng=np_rng,
		mean_span_width=mean_span_width,
		mean_corrupt_chance=mean_corruption_percentage,
		sentinel_start=sentinel_start,
		sentinel_end=sentinel_end,
		max_length_mult=max_length_mult,
		sort_spans=True,
		max_allowed_length=max_allowed_length,
		rng_seed=rng_ctr,
	)
	
	results = corrupt_span_iterative(inputs, **{**span_corruption_kwargs, **pp_kwargs,})

	inputs = results['inputs']

	wd = 100
	if db == 0:
		# hacky to only work 
		sentinel_tokens = np.arange(sentinel_start, sentinel_end)

		has_sentinel = False

		while not has_sentinel:
			for st in sentinel_tokens:
				if st in inputs:
					has_sentinel = True
					break
			
			if has_sentinel:
				break
			
			if not has_sentinel:
				wd -= 1
				if wd <= 0:
					return None

				# print(f"  no sentinels found, retrying... db: {db}, (np.max(inputs): {np.max(inputs)}) (use_autoencoding_objective: {use_autoencoding_objective}),  inputs.shape: {inputs.shape},  labels.shape: {results['labels'].shape},  remaining_space: {remaining_space},  gl: {gl}  ")
				pp_kwargs['rng_seed'] += 1
				pp_kwargs['rng_seed'] = int(pp_kwargs['rng_seed'] ** 2)
				if pp_kwargs['rng_seed'] > 1_000_000_000:
					pp_kwargs['rng_seed'] = -1_000_000_000 

				results = corrupt_span_iterative(inputs, **{**span_corruption_kwargs, **pp_kwargs,})
				inputs = results['inputs']

	return results















if __name__ == "__main__":

	current_element = np.arange(1024)
	remaining_space = 999999
	rng_ctr = 0
	gl = 0
	task_indice = 0

	cc = SpanCorruptionConfig(
		mean_span_width=3,
		mean_corruption_percentage=0.15,
		fitm=0,
		deshuffle=0,
		objective="span_corruption",
		use_autoencoding_objective=False,
	)

	current_instance = SpanCorruptionInstance(
		inputs=current_element,
		max_allowed_length=(remaining_space),
		rng_seed=(rng_ctr*2-1),
		corruption_config=cc,
	)

	rd = find_valid_result(current_instance, use_autoencoding_objective=False, rng_ctr=rng_ctr, corruption_config=cc,)

	if rd is None:
		# forced_task_selection = 0
		current_element = None
		refresh = True
		print(f"  Failed to find a valid result!  ")


	for rd_idx, (key, value) in enumerate(rd.items()):
		print(f"  rd[{key}]: {value}")
		