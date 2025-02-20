"""
Loads an objective based on its name and configuration overrides.

We maintain a registry of known objectives (like autoencoding, infilling, copying, etc.).
"""

from typing import Dict, Any

from birdie_rl.objectives.next_token_prediction import (
	NextTokenPredictionObjective,
	NextTokenPredictionConfig,
)
from birdie_rl.objectives.selective_copying import (
	SelectiveCopyingObjective,
	SelectiveCopyingConfig,
)
from birdie_rl.objectives.copying import CopyingObjective, CopyingConfig
from birdie_rl.objectives.deshuffling import DeshufflingObjective, DeshufflingConfig
from birdie_rl.objectives.infilling import InfillingObjective, InfillingConfig
from birdie_rl.objectives.autoencoding import (
	AutoencodingObjective,
	AutoencodingConfig,
	AutoencodingWithDeshufflingConfig,
)
from birdie_rl.objectives.prefix_language_modeling import (
	PrefixLanguageModelingObjective,
	PrefixLanguageModelingConfig,
)


# Registry that maps objective names to (ObjectiveClass, ConfigClass).
OBJECTIVE_REGISTRY: Dict[str, Any] = {
	"next_token_prediction": (NextTokenPredictionObjective, NextTokenPredictionConfig,),
	"selective_copying": (SelectiveCopyingObjective, SelectiveCopyingConfig),
	"copying": (CopyingObjective, CopyingConfig),
	"deshuffling": (DeshufflingObjective, DeshufflingConfig),
	"infilling": (InfillingObjective, InfillingConfig),
	"autoencoding": (AutoencodingObjective, AutoencodingConfig),
	"autoencoding_with_deshuffling": (AutoencodingObjective, AutoencodingWithDeshufflingConfig),
	"prefix_language_modeling": (PrefixLanguageModelingObjective, PrefixLanguageModelingConfig,),
}


def load_objective(name: str, config_overrides: Dict[str, Any] = None, **kwargs,) -> Any:
	"""
	Load and instantiate an objective by name.

	Args:
		name: Name like "autoencoding", "infilling", etc.
		config_overrides: A dictionary of config overrides.

	Returns:
		An instance of the requested objective.

	Raises:
		ValueError if the objective name is not recognized.
	"""
	if name not in OBJECTIVE_REGISTRY:
		raise ValueError(f"Unknown objective: {name}")

	ObjectiveClass, ConfigClass = OBJECTIVE_REGISTRY[name]

	if config_overrides is None:
		config_overrides = {}

	valid_fields = set(ConfigClass.__dataclass_fields__.keys())
	filtered_overrides = {
		k: v for k, v in config_overrides.items() if k in valid_fields
	}

	config = ConfigClass(**filtered_overrides)
	obj = ObjectiveClass(config)
	return obj


# Demo usage
if __name__ == "__main__":

	import tiktoken
	tokenizer = tiktoken.get_encoding("o200k_base")
	copy_obj = load_objective("copying", {"paradigm": "[COPY]", "tokenizer": tokenizer})
	print("Loaded copying =>", copy_obj)

	ae_obj = load_objective(
		"autoencoding", {"corruption_rate": 0.3, "tokenizer": tokenizer}
	)
	print("Loaded autoencoding =>", ae_obj)