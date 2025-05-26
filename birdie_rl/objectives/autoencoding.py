"""
autoencoding.py

PURPOSE:
  - Defines AutoencodingObjective that randomly masks spans in the text,
    placing placeholders, with the label being the original text or subset.
  - Optimizes by pre-tokenizing static strings like paradigm_prompt, mask_prefix, and mask_suffix.
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

@dataclasses.dataclass
class AutoencodingConfig(BaseObjectiveConfig):
	corruption_rate: float = 0.50
	tokens_per_mask: int = 3
	max_mask_spans: int = 99999
	mask_prefix: str = " [[mask_" # String form for config
	mask_suffix: str = "]]"    # String form for config
	paradigm_prompt: str = "<|AUTOENCODE|>" # String form for config
	max_attempts: int = 100
	separator: str = " " # Not directly tokenized, used in f-string for placeholder
	shuffle: bool = False
	gap_between_spans: int = 1 # Not directly tokenized
	paradigm_end: str = "" # String form for config, if used
	deshuffling_percentage: float = 0.0 # Not directly tokenized

	def __post_init__(self):
		"""
		Adjust derived fields for min/max corruption or mask sizes.
		"""
		self.minimum_corruption_rate = self.corruption_rate * 0.5
		self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
		self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
		self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


@dataclasses.dataclass
class AutoencodingWithDeshufflingConfig(AutoencodingConfig): # Inherits from AutoencodingConfig
	paradigm_prompt: str = "<|AUTOENCODE + DESHUFFLE|>"
	shuffle: bool = True # Default for this variant
	deshuffling_percentage: float = 0.75


class AutoencodingObjective(BaseObjective):
	"""
	Autoencoding objective that inserts placeholders for random spans and
	uses the original text up to max_idx as the label.
	Pre-tokenizes static strings for efficiency.
	"""

	def __init__(self, config: AutoencodingConfig) -> None:
		super().__init__(config)
		# Pre-tokenize static parts
		self.tokenized_paradigm_prompt = []
		if self.config.paradigm_prompt and self.tokenizer:
			self.tokenized_paradigm_prompt = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_prompt))
		
		# Mask prefix/suffix are used within f-strings to generate unique placeholders like "[[mask_0]]"
		# So, we can't pre-tokenize them directly if the number changes.
		# However, if the numeric part was also a fixed token, we could.
		# For now, the placeholder string is generated and then tokenized.
		# If mask_prefix and mask_suffix were always the same (e.g. just MASK_START_TOKEN, MASK_END_TOKEN),
		# then they could be pre-tokenized.
		# The current structure `f"{config.mask_prefix}{placeholders_inserted}{config.mask_suffix}"`
		# means `tokenizer.encode()` will be called for each unique placeholder.
		# This is generally fine as the number of unique placeholders is small per sample.

		self.tokenized_paradigm_end = []
		if hasattr(self.config, 'paradigm_end') and self.config.paradigm_end and self.tokenizer:
			self.tokenized_paradigm_end = self.safe_cast_to_list(self.tokenizer.encode(self.config.paradigm_end))


	def build_input_and_labels(
		self, input_text: str, config: AutoencodingConfig # Config type hint matches base for broader use
	) -> Dict[str, Any]:
		"""
		Build the 'input_ids' with masked spans, and 'label_ids' with the original tokens.
		"""
		tokenizer = self.tokenizer # Should be set from BaseObjective
		encoded_input = tokenizer.encode(input_text)

		prompt_toks = self.tokenized_paradigm_prompt # Use pre-tokenized version

		n_tokens = len(encoded_input)
		# Reserve some space for prompt and potential EOS/padding. Max_n_tokens is for the core text.
		max_n_tokens = min(n_tokens, config.remaining_space - len(prompt_toks) - 16) # -16 is a small buffer
		
		if max_n_tokens <= 0:
			return {
				"status": "fail", "objective": "Autoencoding", "input_ids": [], "label_ids": [],
				"unused_input_string": input_text, "unused_input_ids": encoded_input,
				"masked_count": 0, "original_length": 0,
			}

		def sample_span_length(current_idx_in_text):
			raw_len = self.np_rng.poisson(config.tokens_per_mask)
			raw_len = max(raw_len, config.minimum_tokens_per_mask)
			raw_len = min(raw_len, config.maximum_tokens_per_mask)
			# Ensure span doesn't exceed available text from current_idx_in_text within max_n_tokens
			limit = max_n_tokens - current_idx_in_text 
			return max(1, min(raw_len, limit))


		for attempt_i in range(config.max_attempts):
			current_input_ids_list = list(prompt_toks) # Start with prompt
			placeholders_inserted = 0
			tokens_masked_count = 0
			
			# Index for iterating through the original `encoded_input` up to `max_n_tokens`
			text_idx = 0 
			# Tracks the end of the latest segment from `encoded_input` that has been processed
			# for label generation.
			label_max_original_idx = 0 

			while text_idx < max_n_tokens:
				# Calculate total length so far for input and the potential label extent
				current_total_input_len = len(current_input_ids_list)
				# The label will cover up to `label_max_original_idx` from original text
				# Plus any end paradigm tokens
				potential_total_len = current_total_input_len + label_max_original_idx + len(self.tokenized_paradigm_end)

				if potential_total_len >= config.remaining_space:
					break # Not enough space for more operations

				local_corruption_rate = self.np_rng.uniform(
					config.minimum_corruption_rate, config.maximum_corruption_rate
				)
				span_len_to_mask = sample_span_length(text_idx)
				
				# Probability to mask this span
				# Avoid division by zero if span_len_to_mask is 0 (though sample_span_length ensures >=1)
				prob_to_mask = local_corruption_rate / span_len_to_mask if span_len_to_mask > 0 else 0.0

				if self.np_rng.uniform() < prob_to_mask and (max_n_tokens - text_idx >= span_len_to_mask):
					# Try to mask
					placeholder_str = f"{config.mask_prefix}{placeholders_inserted}{config.mask_suffix}"
					ph_toks = tokenizer.encode(placeholder_str)

					# Check if adding this placeholder and extending label to cover the masked span fits
					if (current_total_input_len + len(ph_toks) + (text_idx + span_len_to_mask) + len(self.tokenized_paradigm_end)) <= config.remaining_space:
						current_input_ids_list.extend(ph_toks)
						label_max_original_idx = max(label_max_original_idx, text_idx + span_len_to_mask)
						text_idx += span_len_to_mask
						placeholders_inserted += 1
						tokens_masked_count += span_len_to_mask
						continue 
					# If not, fall through to adding unmasked token
				
				# Add unmasked token
				if (current_total_input_len + 1 + max(label_max_original_idx, text_idx + 1) + len(self.tokenized_paradigm_end)) <= config.remaining_space:
					current_input_ids_list.append(encoded_input[text_idx])
					label_max_original_idx = max(label_max_original_idx, text_idx + 1)
					text_idx += 1
				else:
					break # No space even for one unmasked token + its label part

			if placeholders_inserted > 0:
				label_ids_list = list(encoded_input[:label_max_original_idx]) # Get original tokens for label
				if self.tokenized_paradigm_end: # Add end paradigm if it exists and fits
				    if len(current_input_ids_list) + len(label_ids_list) + len(self.tokenized_paradigm_end) <= config.remaining_space:
				        label_ids_list.extend(self.tokenized_paradigm_end)


				leftover_original_idx = label_max_original_idx # All original text up to this point is used or accounted for
				unused_input_ids_list = encoded_input[leftover_original_idx:]
				unused_input_str = tokenizer.decode(unused_input_ids_list)

				return {
					"status": "ok", "objective": "Autoencoding",
					"input_ids": np.array(current_input_ids_list, dtype=np.int32),
					"label_ids": np.array(label_ids_list, dtype=np.int32),
					"unused_input_string": unused_input_str,
					"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
					"masked_count": tokens_masked_count,
					"original_length": label_max_original_idx, # Length of original text segment used for labels
				}

		# Fallback if no placeholders inserted after all attempts
		final_input_ids = list(prompt_toks)
		# Use up to max_n_tokens of original text if no masking happened
		final_input_ids.extend(encoded_input[:max_n_tokens]) 
		label_ids_list = list(encoded_input[:max_n_tokens])
		if self.tokenized_paradigm_end:
			if len(final_input_ids) + len(label_ids_list) + len(self.tokenized_paradigm_end) <= config.remaining_space:
				label_ids_list.extend(self.tokenized_paradigm_end)

		unused_input_ids_list = encoded_input[max_n_tokens:]
		unused_input_str = tokenizer.decode(unused_input_ids_list)
		
		return {
			"status": "ok", # Still "ok" but unmasked
			"objective": "Autoencoding (fallback, unmasked)",
			"input_ids": np.array(final_input_ids, dtype=np.int32),
			"label_ids": np.array(label_ids_list, dtype=np.int32),
			"unused_input_string": unused_input_str,
			"unused_input_ids": np.array(unused_input_ids_list, dtype=np.int32),
			"masked_count": 0,
			"original_length": max_n_tokens,
		}
