"""
Prefix Language Modeling objective.

The input is the first X% of tokens (prefix), the label is the remaining tokens (suffix).
Optimized by pre-tokenizing the paradigm_str.
"""

import dataclasses
import numpy as np
from typing import Any, Dict

from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig
from birdie_rl.objectives.utils import slice_text_by_remaining_space


@dataclasses.dataclass
class PrefixLanguageModelingConfig(BaseObjectiveConfig):
	"""
	Configuration for prefix LM.

	Attributes:
		prefix_fraction: Fraction of tokens in prefix (0.0 to 1.0).
		paradigm_str: An optional string to prepend to the prefix tokens.
		# special_token_ids: Optional dict of special tokens (unused in this example).
	"""
	prefix_fraction: float = 0.75
	paradigm_str: str = "<|PREFIX LM|>" # String form for config
	# special_token_ids: dict = dataclasses.field(default_factory=dict) # Not used


class PrefixLanguageModelingObjective(BaseObjective):
	"""
	For prefix LM: 
	  - The input is [paradigm_str_tokens] + the first `prefix_fraction` of text tokens.
	  - The label is the remaining text tokens.
	Pre-tokenizes paradigm_str.
	"""

	def __init__(self, config: PrefixLanguageModelingConfig) -> None:
		super().__init__(config)
		# Pre-tokenize static parts
		self.tokenized_paradigm_str = []
		if self.config.paradigm_str and self.tokenizer:
			self.tokenized_paradigm_str = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_str))

	def build_input_and_labels(
		self, input_text: str, config: PrefixLanguageModelingConfig
	) -> Dict[str, Any]:
		"""
		Construct the prefix as input and the suffix as label.
		"""
		paradigm_tokens_list = self.tokenized_paradigm_str # Use pre-tokenized

		# Calculate space available for the actual text (prefix + suffix)
		space_for_text = config.remaining_space - len(paradigm_tokens_list)
		
		if space_for_text < 2: # Need at least 1 for prefix, 1 for suffix (or handle differently)
			return { # Not enough space for a meaningful prefix/suffix split
				"status": "fail", "objective": "Prefix Language Modeling",
				"input_ids": np.array(paradigm_tokens_list, dtype=np.int32), # Only paradigm if it fits
				"label_ids": np.array([], dtype=np.int32),
				"unused_input_string": input_text, 
				"unused_input_ids": np.array(self.tokenizer.encode(input_text), dtype=np.int32),
				"message": "Not enough space for text after paradigm."
			}
			
		slice_data = slice_text_by_remaining_space(
			text=input_text,
			tokenizer=self.tokenizer,
			remaining_space=space_for_text, # Use adjusted space
		)
		
		used_tokens_list = self.safe_cast_to_list(slice_data["used_tokens"])
		leftover_text = slice_data["unused_text"]
		leftover_tokens_list = self.safe_cast_to_list(slice_data["unused_tokens"])

		length_of_used_text_tokens = len(used_tokens_list)

		if length_of_used_text_tokens < 2: # e.g., need at least one for prefix, one for suffix
			# Fallback: input is paradigm + all used_tokens, label is all used_tokens (or empty if no paradigm makes it too long)
			# This effectively becomes next token prediction if prefix is empty.
			final_input_ids_list = list(paradigm_tokens_list) + used_tokens_list
			final_label_ids_list = used_tokens_list
			
			# Final check if this fallback fits
			if len(final_input_ids_list) + len(final_label_ids_list) > config.remaining_space:
				# If even fallback doesn't fit, return minimal or error
				return {
					"status": "fail", "objective": "Prefix Language Modeling",
					"input_ids": np.array(paradigm_tokens_list, dtype=np.int32),
					"label_ids": np.array([], dtype=np.int32),
					"unused_input_string": input_text, "unused_input_ids": np.array(self.tokenizer.encode(input_text), dtype=np.int32),
					"message": "Fallback (NTP style) for very short text also too long."
				}

			return {
				"status": "ok", "objective": "Prefix Language Modeling (short text fallback)",
				"input_ids": np.array(final_input_ids_list, dtype=np.int32),
				"label_ids": np.array(final_label_ids_list, dtype=np.int32),
				"unused_input_string": leftover_text,
				"unused_input_ids": np.array(leftover_tokens_list, dtype=np.int32),
			}

		prefix_len = int(np.floor(length_of_used_text_tokens * config.prefix_fraction))
		# Ensure prefix_len is at least 1 if possible, and suffix also has at least 1 if possible
		prefix_len = max(1, min(prefix_len, length_of_used_text_tokens - 1)) 
		
		prefix_toks_list = used_tokens_list[:prefix_len]
		suffix_toks_list = used_tokens_list[prefix_len:]

		final_input_ids_list = list(paradigm_tokens_list) + prefix_toks_list
		final_label_ids_list = suffix_toks_list # Label is just the suffix

		# This check should ideally be redundant due to `space_for_text` and slicing,
		# but as a safeguard:
		if len(final_input_ids_list) + len(final_label_ids_list) > config.remaining_space:
			return {
				"status": "error", "objective": "Prefix Language Modeling",
				"message": (
					f"Combined length {len(final_input_ids_list) + len(final_label_ids_list)} "
					f"exceeds remaining_space={config.remaining_space}. Should not happen due to pre-slicing."
				),
			}

		return {
			"status": "ok", "objective": "Prefix Language Modeling",
			"input_ids": np.array(final_input_ids_list, dtype=np.int32),
			"label_ids": np.array(final_label_ids_list, dtype=np.int32),
			"unused_input_string": leftover_text,
			"unused_input_ids": np.array(leftover_tokens_list, dtype=np.int32),
		}


# Example demo
if __name__ == "__main__":
	try:
		from birdie_rl.modeling.tokenizer import Tokenizer 
	except ImportError:
		class Tokenizer: # Mock tokenizer
			def encode(self, t): return [ord(c) for c in t] if isinstance(t, str) else [[ord(c) for c in s] for s in t]
			def decode(self, ids): 
				if not ids: return ""
				if isinstance(ids[0], list): return ["".join([chr(i) for i in id_list if i >=0]) for id_list in ids]
				return "".join([chr(i) for i in ids if i >=0])

	tok = Tokenizer()
	text = "This is a test for prefix language modeling. " * 3

	cfg = PrefixLanguageModelingConfig(
		remaining_space=40,
		prefix_fraction=0.7,
		paradigm_str="<<PREFIX>> ",
		tokenizer=tok # Pass tokenizer in config
	)
	obj = PrefixLanguageModelingObjective(cfg)
	# obj.set_tokenizer(tok) # Not needed if passed in config
	result = obj(text)

	print("\n--- Prefix LM Demo ---")
	print("Status:", result["status"])
	if result["status"] == "ok":
		print("Original text (short):", text[:60], "...")
		print("Input IDs (len {}):".format(len(result["input_ids"])), result["input_ids"])
		print("Decoded Input:", tok.decode(result["input_ids"].tolist()))
		print("Label IDs (len {}):".format(len(result["label_ids"])), result["label_ids"])
		print("Decoded Label:", tok.decode(result["label_ids"].tolist()))
		print("Unused text:", result["unused_input_string"])
	else:
		print("Error/Message:", result.get("message", "N/A"))

