"""
Next Token Prediction objective.
"""

import dataclasses
import numpy as np
from typing import Any, Dict

from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig
from birdie_rl.objectives.utils import slice_text_by_remaining_space


@dataclasses.dataclass
class NextTokenPredictionConfig(BaseObjectiveConfig):
	"""
	Configuration for Next Token Prediction.

	Attributes:
		special_token_ids: An optional dict of special tokens.
		use_begin_generating_paradigm: If True, might add something like <BOS>.
		paradigm: Optional prefix text for the input.
	"""

	special_token_ids: dict = dataclasses.field(default_factory=dict)
	use_begin_generating_paradigm: bool = False
	paradigm: str = "<|NTP|>"


class NextTokenPredictionObjective(BaseObjective):
	"""
	Next Token Prediction: the label is the raw text, possibly
	with an added 'paradigm' prefix in the input.
	"""

	def __init__(self, config: NextTokenPredictionConfig) -> None:
		super().__init__(config)

		self.tokenized_paradigm = self.safe_cast_to_list(self.tokenizer.encode(config.paradigm))

	def build_input_and_labels(
		self, input_text: str, config: NextTokenPredictionConfig
	) -> Dict[str, Any]:
		"""
		Build the final dictionary for Next Token Prediction.
		"""
		# paradigm_toks = self.tokenizer.encode(config.paradigm).tolist()
		paradigm_toks = self.tokenized_paradigm
		slice_data = slice_text_by_remaining_space(
			text=input_text,
			tokenizer=self.tokenizer,
			remaining_space=config.remaining_space - len(paradigm_toks),
		)
		used_tokens = slice_data["used_tokens"]
		leftover_text = slice_data["unused_text"]
		leftover_tokens = slice_data["unused_tokens"]

		final_input_tokens = []

		if config.paradigm:
			final_input_tokens.extend(paradigm_toks)

		final_input_ids = np.array(final_input_tokens, dtype=np.int32)
		label_ids = np.array(used_tokens, dtype=np.int32)

		return {
			"status": "ok",
			"objective": "Next Token Prediction",
			"input_ids": final_input_ids,
			"label_ids": label_ids,
			"unused_input_string": leftover_text,
			"unused_input_ids": leftover_tokens,
		}
	

if __name__ == "__main__":
	from modeling.tokenizer import Tokenizer

	tok = Tokenizer()
	text = "Hello, this is a next token prediction test. " * 3
	cfg = NextTokenPredictionConfig(
		remaining_space=40, paradigm="(NTP Prefix) ", use_begin_generating_paradigm=True
	)
	obj = NextTokenPredictionObjective(cfg)
	obj.set_tokenizer(tok)
	result = obj(text)

	print("\n=== Next Token Prediction Demo ===")
	print("Original text (short):", text[:60], "...")
	print("Input IDs:", result["input_ids"])
	print("Decoded Input:", tok.decode(result["input_ids"]))
	print("Label IDs:", result["label_ids"])
	print("Decoded Label:", tok.decode(result["label_ids"]))
	print("Unused text:", result["unused_input_string"])
