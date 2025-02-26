"""
Infilling Objective:
-------------------
- Places placeholders for certain spans in the input (decided stochastically).
- The label is built by concatenating [placeholder + masked tokens] for each span.
- We skip any span if adding it would exceed `remaining_space`.
- If no spans are inserted after `max_attempts`, we revert to returning unmasked text.

Important Logic Points:
1) We do NOT subtract the number of prompt tokens from our text budget. Instead,
   we rely on "prospective" checks to ensure (input + label) ≤ remaining_space.
2) Each iteration, we try either:
   - mask a span (with probability ~ local_corruption_rate/span_len),
   - or pass through 1 unmasked token.
3) We do multiple attempts (max_attempts). If any attempt results in at least 1 placeholder,
   we finalize that. Else, after all attempts, fallback to unmasked text.

Usage:
  obj = InfillingObjective(some_config)
  obj.set_tokenizer(your_tokenizer)
  result = obj("Your text here")

Result Dictionary Keys:
  - "status": "ok"
  - "input_ids": ...
  - "label_ids": ...
  - "unused_input_string": leftover text (if any tokens remain un-used)
  - "unused_input_ids": leftover tokens
  - "masked_count": how many tokens got masked
  - "original_length": how many tokens from the original text we used
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

@dataclasses.dataclass
class InfillingConfig(BaseObjectiveConfig):
	corruption_rate: float = 0.15
	mean_tokens_per_span: int = 3
	max_mask_spans: int = 20
	mask_prefix: str = " [[mask_"
	mask_suffix: str = "]]"
	shuffle: bool = False
	separator: str = " "
	paradigm: str = "<|INFILL|>"
	gap_between_spans: int = 1
	max_attempts: int = 100
	paradigm_end: str = ""
	# minimum_remaining_space: int = 256
	# minimum_sequence_length: int = 256

	def __post_init__(self):
		"""
		Adjusts some derived fields for min/max corruption rates
		and mean_tokens_per_span range.
		"""
		assert(0.0 <= self.corruption_rate <= 1.0)
		self.minimum_corruption_rate = max(0.0, self.corruption_rate * 0.5)
		self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
		self.minimum_mean_tokens_per_span = max(1, self.mean_tokens_per_span // 3)
		self.maximum_mean_tokens_per_span = max(1, self.mean_tokens_per_span * 3)


class InfillingObjective(BaseObjective):
	"""
	InfillingObjective (final, with commented-out debug prints).

	Overview:
	  1) Takes text + optional paradigm => merges them in `input_ids`.
	  2) Iterates tokens, deciding whether to mask a span or pass a single token unmasked.
	  3) Each masked span => we place the placeholder in input_ids, and
		 we append [placeholder + masked_tokens] to label_ids.
	  4) If we insert at least 1 placeholder, we finalize; else we attempt again up to max_attempts.
	  5) If still no placeholders => return unmasked.
	"""

	def __init__(self, config: InfillingConfig) -> None:
		super().__init__(config)

	def build_input_and_labels(
		self, input_text: str, config: InfillingConfig
	) -> Dict[str, Any]:
		"""
		Build "input_ids" (with placeholders) and "label_ids" ([placeholder + tokens])
		according to Infilling logic.

		Args:
			input_text: Raw string to be infilled.
			config: InfillingConfig specifying corruption rates, mean_tokens_per_span, etc.

		Returns:
			A dict with fields:
			 - status
			 - objective
			 - input_ids
			 - label_ids
			 - unused_input_string
			 - unused_input_ids
			 - masked_count
			 - original_length
		"""
		# # (Optional) Debug or info messages:
		# print(f"[InfillingObjective] build_input_and_labels() -> text len = {len(input_text)}, remaining={config.remaining_space}")

		# Encode the text and optional prompt
		tokenizer = self.tokenizer
		encoded_input = tokenizer.encode(input_text)

		prompt_toks = []
		if config.paradigm:
			prompt_toks = tokenizer.encode(config.paradigm)

		# Decide how many tokens from the original text we can consider
		n_tokens = len(encoded_input)
		max_n_tokens = min(n_tokens, config.remaining_space)
		if max_n_tokens <= 0:
			print("[InfillingObjective] No space -> returning empty.")
			return {
				"status": "fail",
				"objective": "Infilling",
				"input_ids": [],
				"label_ids": [],
				"unused_input_string": input_text,
				"unused_input_ids": encoded_input,
				"masked_count": 0,
				"original_length": 0,
			}

		# Function to determine how many tokens we mask in one span
		def sample_span_length(start_idx: int) -> int:
			"""
			Poisson-based sampling for number of tokens in a masked span, clamped by config.
			"""
			raw_len = self.np_rng.poisson(config.mean_tokens_per_span)
			raw_len = max(raw_len, config.minimum_mean_tokens_per_span)
			raw_len = min(raw_len, config.maximum_mean_tokens_per_span)
			# Also ensure we don't exceed max_n_tokens from the current index
			limit = max_n_tokens - start_idx
			if raw_len > limit:
				raw_len = limit
			return max(raw_len, 1)

		# Try multiple attempts to insert at least one placeholder
		for attempt_i in range(config.max_attempts):
			# # print(f"  [Attempt {attempt_i+1}/{config.max_attempts}]")
			input_ids = []
			label_blocks = []
			placeholders_inserted = 0
			masked_tokens_count = 0

			# Add the prompt tokens at the beginning
			input_ids.extend(prompt_toks)

			idx = 0
			while idx < max_n_tokens:
				in_len = len(input_ids)
				lbl_len = sum(len(b) for b in label_blocks)
				total_so_far = in_len + lbl_len

				# If we have no leftover space for even 1 token, break
				if total_so_far >= config.remaining_space:
					# # print("  => No leftover space for any token -> break.")
					break

				# Compute local corruption rate and a span length
				local_corruption_rate = self.np_rng.uniform(
					config.minimum_corruption_rate,
					config.maximum_corruption_rate
				)
				span_len = sample_span_length(idx)
				# Probability that we actually do a mask here
				p = local_corruption_rate / span_len
				# # print(f"   idx={idx}, span_len={span_len}, p={p}")

				# Attempt to mask with probability p
				if self.np_rng.uniform() < p:
					snippet = encoded_input[idx : idx + span_len]
					if len(snippet) > 0:
						# Build placeholder text
						ph_str = f"{config.mask_prefix}{placeholders_inserted}{config.mask_suffix}"
						ph_toks = tokenizer.encode(ph_str)

						# Check prospective size
						prospective_in_len  = in_len + len(ph_toks)
						prospective_lbl_len = lbl_len + len(ph_toks) + span_len
						prospective_total   = prospective_in_len + prospective_lbl_len
						if prospective_total <= config.remaining_space:
							# Accept => place placeholder in input
							input_ids.extend(ph_toks)
							# Then [placeholder + snippet] in label
							block = list(ph_toks) + list(snippet)
							label_blocks.append(block)
							placeholders_inserted += 1
							masked_tokens_count += span_len
							idx += span_len
							continue
					# If snippet empty or can't fit, do not mask => fallback below

				# If not masking => add 1 token unmasked
				prospective_in_len = in_len + 1
				prospective_total  = prospective_in_len + lbl_len
				if prospective_total > config.remaining_space:
					# # print("  => Can't fit 1 unmasked token, break.")
					break

				input_ids.append(encoded_input[idx])
				idx += 1

			# If we inserted placeholders, finalize
			if placeholders_inserted > 0:
				# # print(f"  => placeholders_inserted={placeholders_inserted}, masked={masked_tokens_count}, finalize.")
				label_ids = []
				for block in label_blocks:
					label_ids.extend(block)

				# Possibly add config.paradigm_end if space
				end_toks = tokenizer.encode(config.paradigm_end)
				final_in_len  = len(input_ids)
				final_lbl_len = len(label_ids)
				if (final_in_len + final_lbl_len + len(end_toks)) <= config.remaining_space:
					label_ids.extend(end_toks)
					# # print("   => appended paradigm_end in label_ids")

				leftover_ids = encoded_input[idx:]
				leftover_text = tokenizer.decode(leftover_ids)
				return {
					"status": "ok",
					"objective": "Infilling",
					"input_ids": input_ids,
					"label_ids": label_ids,
					"unused_input_string": leftover_text,
					"unused_input_ids": leftover_ids,
					"masked_count": masked_tokens_count,
					"original_length": max_n_tokens,
				}

			# # print("  => placeholders_inserted=0, continuing attempts...")

		# If we exit after all attempts => no placeholders
		# # print("[InfillingObjective] => no placeholders => returning unmasked text.")
		used_ids = encoded_input[:max_n_tokens]
		leftover_ids = encoded_input[max_n_tokens:]
		leftover_text = tokenizer.decode(leftover_ids)
		return {
			"status": "ok",
			"objective": "Infilling",
			"input_ids": self.safe_cast_to_list(used_ids),
			"label_ids": [],
			"unused_input_string": leftover_text,
			"unused_input_ids": leftover_ids,
			"masked_count": 0,
			"original_length": max_n_tokens,
		}


# Example standalone usage test
if __name__ == "__main__":
	from birdie_rl.modeling.tokenizer import Tokenizer

	tok = Tokenizer()
	text = "Final test for infilling with minimal debug prints."
	cfg = InfillingConfig(
		remaining_space=30,
		corruption_rate=0.15,
		mean_tokens_per_span=3,
		max_attempts=5,
		mask_prefix="[mask_",
		paradigm="<INFILL_FINAL>",
	)
	obj = InfillingObjective(cfg)
	obj.set_tokenizer(tok)

	result = obj(text)
	print("Status =", result["status"])
	print("Decoded input =", tok.decode(result["input_ids"]))
	print("Decoded label =", tok.decode(result["label_ids"]))
	print("Unused text =", result["unused_input_string"])
