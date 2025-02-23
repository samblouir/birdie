"""
Base classes for objectives.

Defines:
- BaseObjectiveConfig (dataclass) for objective settings.
- BaseObjective (abstract class) which each objective implements.
"""

from typing import Any, Dict, List, Union
import dataclasses
import copy
import numpy as np
from birdie_rl.objectives import utils

@dataclasses.dataclass
class BaseObjectiveConfig:
	"""
	Base configuration class for objectives.

	Attributes:
		objective: A descriptive name of the objective (optional).
		rng_seed: Random seed for the objective's RNG.
		minimum_sequence_length: Enforced minimum input length.
		maximum_sequence_length: Enforced maximum input length.
		minimum_remaining_space: The minimum required remaining space to operate.
		maximum_remaining_space: The maximum allowed remaining space to operate.
		remaining_space: The actual space left, passed in externally.
		tokenizer: The tokenizer instance to use. If None, a default is created.
	"""

	objective: str = ""
	rng_seed: int = 42
	# minimum_sequence_length: int = -1
	minimum_sequence_length: int = 32
	minimum_remaining_space: int = 32
	maximum_sequence_length: int = -1
	maximum_remaining_space: int = -1
	remaining_space: int = -1
	tokenizer: Any = None


class BaseObjective:
	"""
	Abstract base class for objectives.

	Subclasses should override `build_input_and_labels` to produce
	the final "input_ids", "label_ids", and other metadata.
	"""

	def __init__(self, config: BaseObjectiveConfig) -> None:
		"""
		Initialize the objective with the given configuration.

		Args:
			config: An instance of BaseObjectiveConfig (or subclass).
		"""
		self.config = config
		self.np_rng = np.random.default_rng(self.config.rng_seed)

		if self.config.tokenizer is None:
			raise Exception("Please provide a Tokenizer in the BaseObjectiveConfig before creating a BaseObjective.")
		else:
			self.set_tokenizer(self.config.tokenizer)

	def hash(self):
		"""
		Return a unique hash for the objective.
		"""
		config_without_keys = {k: v for k, v in self.config.__dict__.items() if k not in ["tokenizer"]}
		return utils.sha_hash(config_without_keys)

	def set_tokenizer(self, tokenizer: Any) -> None:
		"""
		Set the tokenizer for the objective.

		Args:
			tokenizer: The tokenizer instance to use.
		"""
		self.tokenizer = tokenizer
		self.config.tokenizer = tokenizer

	def __call__(
		self, input_ids: Union[str, List[int]], **kwargs
	) -> Dict[str, Any]:
		"""
		Make the objective callable, e.g. objective(text).

		Args:
			input_ids: Input text (string) or token IDs (list).
			**kwargs: Additional config overrides.

		Returns:
			A dictionary with status, input_ids, label_ids, etc.
		"""
		merged_config = copy.deepcopy(self.config)
		for key, val in kwargs.items():
			if hasattr(merged_config, key):
				setattr(merged_config, key, val)

		self.np_rng = np.random.default_rng(merged_config.rng_seed)
		check_status = self.run_checks(input_ids, merged_config)

		if check_status["status"] != "ok":
			return check_status

		result = self.build_input_and_labels(input_ids, merged_config)

		if merged_config.maximum_sequence_length >= 0:
			final_length = len(result["input_ids"])
			if final_length != merged_config.maximum_sequence_length:
				return {
					"status": "error",
					"message": (
						f"Input sequence too long: final input_ids length "
						f"{final_length} does not equal required maximum "
						f"{merged_config.maximum_sequence_length}"
					),
				}

		return result

	def run_checks(
		self, input_ids: Union[str, List[int]], config: BaseObjectiveConfig
	) -> Dict[str, Any]:
		"""
		Perform validation checks on the input and configuration.

		Args:
			input_ids: The raw input (text or token IDs).
			config: The configuration for the objective.

		Returns:
			A dict with "status" = "ok" or "error", plus a message if error.
		"""
		if isinstance(input_ids, (list, np.ndarray)):
			length = len(input_ids)
		else:
			length = len(self.tokenizer.encode(input_ids))

		if config.minimum_sequence_length >= 0 and length < config.minimum_sequence_length:
			return {
				"status": "error",
				"message": (
					f"Input sequence too short: {length} < min "
					f"{config.minimum_sequence_length}"
				),
			}

		if config.maximum_sequence_length >= 0 and length > config.maximum_sequence_length:
			return {
				"status": "error",
				"message": (
					f"Input sequence too long: {length} > max "
					f"{config.maximum_sequence_length}"
				),
			}

		if config.minimum_remaining_space >= 0 and config.remaining_space >= 0:
			if config.remaining_space < config.minimum_remaining_space:
				return {
					"status": "error",
					"message": (
						f"Not enough space left: config.remaining_space: {config.remaining_space} < min config.minimum_remaining_space: ({config.minimum_remaining_space})"
					),
				}

		if config.maximum_remaining_space >= 0 and config.remaining_space >= 0:
			if config.remaining_space > config.maximum_remaining_space:
				return {
					"status": "error",
					"message": (
						f"Remaining space {config.remaining_space} exceeds "
						f"maximum allowed {config.maximum_remaining_space}"
					),
				}

		return {"status": "ok"}

	def build_input_and_labels(
		self, input_ids: Union[str, List[int]], config: BaseObjectiveConfig
	) -> Dict[str, Any]:
		"""
		Subclasses must implement this method.

		It should return a dict with 'input_ids', 'label_ids', etc.
		"""
		raise NotImplementedError("Subclasses must implement build_input_and_labels.")

	def safe_cast_to_list(self, x):
		try:
			return x.tolist()
		except:
			return x


if __name__ == "__main__":

	class DummyObjective(BaseObjective):
		def build_input_and_labels(self, input_ids, config):
			return {"status": "ok", "dummy": True}

	dummy_cfg = BaseObjectiveConfig()
	dummy_obj = DummyObjective(dummy_cfg)
	result = dummy_obj([1, 2, 3])
	print("Dummy objective call =>", result)
