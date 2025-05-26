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
		# Pre-tokenize the paradigm string during initialization
		if self.config.paradigm:
			self.tokenized_paradigm = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm))
		else:
			self.tokenized_paradigm = []


	def build_input_and_labels(
		self, input_text: str, config: NextTokenPredictionConfig
	) -> Dict[str, Any]:
		"""
		Build the final dictionary for Next Token Prediction.
		Uses the pre-tokenized paradigm.
		"""
		# Use the pre-tokenized paradigm from __init__
		paradigm_toks = self.tokenized_paradigm

		# Calculate remaining space for the main text, after accounting for paradigm tokens
		space_for_text = config.remaining_space - len(paradigm_toks)
		
		slice_data = slice_text_by_remaining_space(
			text=input_text,
			tokenizer=self.tokenizer,
			remaining_space=space_for_text, # Use adjusted remaining space
		)
		
		used_tokens = slice_data["used_tokens"]
		leftover_text = slice_data["unused_text"]
		leftover_tokens = slice_data["unused_tokens"]

		final_input_tokens_list = []

		if paradigm_toks: # Check if there are paradigm tokens to add
			final_input_tokens_list.extend(paradigm_toks)
		
		# Add the used tokens from the text if any
		if len(used_tokens) > 0:
			final_input_tokens_list.extend(self.safe_cast_to_list(used_tokens))


		# Ensure final_input_ids and label_ids are NumPy arrays
		# If final_input_tokens_list is empty, make an empty array, otherwise convert
		final_input_ids = np.array(final_input_tokens_list, dtype=np.int32) if final_input_tokens_list else np.array([], dtype=np.int32)
		
		# label_ids are just the used_tokens (or empty if none were used)
		label_ids = np.array(used_tokens, dtype=np.int32) if len(used_tokens) > 0 else np.array([], dtype=np.int32)


		return {
			"status": "ok",
			"objective": "Next Token Prediction",
			"input_ids": final_input_ids,
			"label_ids": label_ids,
			"unused_input_string": leftover_text,
			"unused_input_ids": leftover_tokens,
		}
	

if __name__ == "__main__":
	# This import is problematic if modeling.tokenizer is not in PYTHONPATH
	# For standalone testing, you might need to adjust imports or use a mock tokenizer
	try:
		from modeling.tokenizer import Tokenizer 
	except ImportError:
		# Fallback for environments where birdie_rl is not installed in a way that resolves this
		# This is a common issue when running files directly from within a package subfolder.
		# A simple mock tokenizer for testing purposes:
		class Tokenizer:
			def encode(self, text_batch, **kwargs):
				if isinstance(text_batch, str):
					return [ord(c) for c in text_batch]
				elif isinstance(text_batch, list):
					return [[ord(c) for c in text] for text in text_batch]
				return []
			def decode(self, tokens_batch, **kwargs):
				if not tokens_batch: return ""
				if isinstance(tokens_batch[0], list): # Batch of token lists
					return ["".join([chr(t) for t in tokens if t >= 0]) for tokens in tokens_batch]
				else: # Single token list
					return "".join([chr(t) for t in tokens_batch if t >= 0])


	tok = Tokenizer()
	text = "Hello, this is a next token prediction test. " * 3
	cfg = NextTokenPredictionConfig(
		remaining_space=40, paradigm="(NTP Prefix) ", use_begin_generating_paradigm=True, tokenizer=tok
	)
	obj = NextTokenPredictionObjective(cfg)
	# obj.set_tokenizer(tok) # Tokenizer is now set in config and passed to super().__init__
	result = obj(text)

	print("\n=== Next Token Prediction Demo ===")
	print("Original text (short):", text[:60], "...")
	print("Input IDs:", result["input_ids"])
	print("Decoded Input:", tok.decode(result["input_ids"]))
	print("Label IDs:", result["label_ids"])
	print("Decoded Label:", tok.decode(result["label_ids"]))
	print("Unused text:", result["unused_input_string"])
