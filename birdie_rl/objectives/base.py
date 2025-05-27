"""
Base classes for objectives.

Defines:
- BaseObjectiveConfig (dataclass) for objective settings.
- BaseObjective (abstract class) which each objective implements.
- run_checks now returns a specific "not_enough_space" status.
"""

from typing import Any, Dict, List, Union
import dataclasses
import copy 
import numpy as np
from birdie_rl.objectives import utils
import sys # For sys.stdout.flush() in example

@dataclasses.dataclass
class BaseObjectiveConfig:
	objective: str = ""
	rng_seed: int = 42
	minimum_sequence_length: int = 128
	minimum_remaining_space: int = 256 # Min space an objective needs to even attempt processing
	maximum_sequence_length: int = -1
	maximum_remaining_space: int = -1
	remaining_space: int = -1 # Actual space in current packer passed by Worker
	tokenizer: Any = None 


class BaseObjective:
	def __init__(self, config: BaseObjectiveConfig) -> None:
		self.config = config 
		
		if self.config.tokenizer is None:
			raise ValueError("Tokenizer must be provided in the BaseObjectiveConfig passed to BaseObjective.")
		self.tokenizer = self.config.tokenizer 

		self.np_rng = np.random.default_rng(self.config.rng_seed)


	def hash(self):
		config_dict_for_hash = {
			k: v for k, v in self.config.__dict__.items() 
			if k != "tokenizer" and not callable(v) 
		}
		return utils.sha_hash(config_dict_for_hash)

	def set_tokenizer(self, tokenizer: Any) -> None:
		self.tokenizer = tokenizer
		if hasattr(self.config, 'tokenizer'):
			self.config.tokenizer = tokenizer

	def __call__(
		self, input_ids: Union[str, List[int]] 
	) -> Dict[str, Any]:
		if hasattr(self.config, 'rng_seed'): # Re-seed if objective instance is cached and reused
			self.np_rng = np.random.default_rng(self.config.rng_seed)

		# run_checks uses self.config, which includes remaining_space set by Worker
		check_status = self.run_checks(input_ids, self.config) 

		if check_status["status"] != "ok":
			return check_status # Propagate specific error status like "not_enough_space"

		result = self.build_input_and_labels(input_ids, self.config) 
		return result

	def run_checks(
		self, input_ids: Union[str, List[int]], config: BaseObjectiveConfig
	) -> Dict[str, Any]:
		if self.tokenizer is None: 
			return {"status": "error", "message": "Tokenizer not available for run_checks (self.tokenizer is None)."}

		# Check for minimum_remaining_space first, as this is a prerequisite for an objective to operate
		if hasattr(config, 'remaining_space') and config.remaining_space >=0 :
			if hasattr(config, 'minimum_remaining_space') and config.minimum_remaining_space > 0: # Ensure it's a positive requirement
				if config.remaining_space < config.minimum_remaining_space:
					return {
						"status": "not_enough_space", # Specific status
						"message": (
							f"Not enough space left for objective to run: config.remaining_space ({config.remaining_space}) "
							f"< objective's min_remaining_space ({config.minimum_remaining_space})"
						),
					}
		# Length checks for the input_ids itself
		if isinstance(input_ids, (list, np.ndarray)):
			length = len(input_ids)
		elif isinstance(input_ids, str): 
			length = len(self.tokenizer.encode(input_ids)) # This could be expensive if called often before actual processing
		else:
			return {"status": "error", "message": f"Invalid input_ids type: {type(input_ids)}"}

		if hasattr(config, 'minimum_sequence_length') and config.minimum_sequence_length >= 0 and length < config.minimum_sequence_length:
			return {
				"status": "error", # Or perhaps "input_too_short"
				"message": (
					f"Input sequence too short: {length} < min "
					f"{config.minimum_sequence_length}"
				),
			}

		if hasattr(config, 'maximum_sequence_length') and config.maximum_sequence_length >= 0 and length > config.maximum_sequence_length:
			return {
				"status": "error", # Or "input_too_long"
				"message": (
					f"Input sequence too long: {length} > max "
					f"{config.maximum_sequence_length}"
				),
			}
		
		# Redundant check for maximum_remaining_space, usually remaining_space is what matters for fitting output.
		# if hasattr(config, 'maximum_remaining_space') and config.maximum_remaining_space >= 0:
		# 	if config.remaining_space > config.maximum_remaining_space:
		# 		return {
		# 			"status": "error",
		# 			"message": (
		# 				f"Remaining space {config.remaining_space} exceeds "
		# 				f"maximum allowed {config.maximum_remaining_space}"
		# 			),
		# 		}

		return {"status": "ok"}

	def build_input_and_labels(
		self, input_ids: Union[str, List[int]], config: BaseObjectiveConfig
	) -> Dict[str, Any]:
		raise NotImplementedError("Subclasses must implement build_input_and_labels.")

	def safe_cast_to_list(self, x):
		if isinstance(x, np.ndarray):
			return x.tolist()
		elif isinstance(x, list):
			return x
		elif x is None:
			return []
		try:
			return list(x)
		except TypeError: 
			return [x]


if __name__ == "__main__":

	@dataclasses.dataclass
	class DummyObjectiveConfig(BaseObjectiveConfig):
		my_param: str = "default"
		minimum_remaining_space: int = 10 # Override base for test

	class DummyObjective(BaseObjective):
		def __init__(self, config: DummyObjectiveConfig): 
			super().__init__(config)
			if self.tokenizer and hasattr(self.config, 'my_param'):
				print(f"DummyObjective initialized with tokenizer and param: {self.config.my_param}", flush=True)

		def build_input_and_labels(self, input_ids, config: DummyObjectiveConfig): 
			# Simulate using some space
			if config.remaining_space < 20: # Arbitrary check for this dummy
				# This specific objective might decide it can't produce good output
				# but this is different from the BaseObjective.run_checks
				# For this test, assume it always works if run_checks passed.
				pass
			return {"status": "ok", "dummy": True, "param_used": config.my_param, "seed_used": config.rng_seed, "input_ids":[1,2,3], "label_ids":[1,2,3]}

	class MockTokenizer:
		def encode(self, t): return [ord(c) for c in t] if isinstance(t, str) else []
		def decode(self, ids): return "".join([chr(i) for i in ids if i >=0])

	mock_tokenizer = MockTokenizer()
	
	# Test "not_enough_space"
	cfg_not_enough_space = DummyObjectiveConfig(tokenizer=mock_tokenizer, remaining_space=5) # remaining_space < minimum_remaining_space (10)
	obj_nes = DummyObjective(cfg_not_enough_space)
	result_nes = obj_nes("short text") # __call__ will run checks
	print("Result (Not Enough Space):", result_nes, flush=True)
	assert result_nes["status"] == "not_enough_space"

	# Test "ok"
	cfg_ok_space = DummyObjectiveConfig(tokenizer=mock_tokenizer, remaining_space=50, my_param="OK_PARAM", rng_seed=777)
	obj_ok = DummyObjective(cfg_ok_space)
	result_ok = obj_ok("some good text")
	print("Result (OK Space):", result_ok, flush=True)
	assert result_ok["status"] == "ok"
	assert result_ok["param_used"] == "OK_PARAM"
	assert result_ok["seed_used"] == 777

