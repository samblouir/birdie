"""
autoencoding.py

PURPOSE:
  - Defines AutoencodingObjective that randomly masks spans in the text,
    placing placeholders, with the label being the original text or subset.

USAGE:
  from birdie_rl.objectives.autoencoding import AutoencodingObjective, AutoencodingConfig
  obj = AutoencodingObjective(AutoencodingConfig(corruption_rate=0.2, tokens_per_mask=3))
  obj.set_tokenizer(tokenizer)
  result = obj("Some text here")
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
    mask_prefix: str = " [[mask_"
    mask_suffix: str = "]]"
    paradigm_prompt: str = "<|AUTOENCODE|>"
    max_attempts: int = 100
    separator: str = " "
    shuffle: bool = False
    gap_between_spans: int = 1
    paradigm_end: str = ""
    deshuffling_percentage: float = 0.0

    def __post_init__(self):
        """
        Adjust derived fields for min/max corruption or mask sizes.
        """
        self.minimum_corruption_rate = self.corruption_rate * 0.5
        self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
        self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
        self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


@dataclasses.dataclass
class AutoencodingWithDeshufflingConfig(BaseObjectiveConfig):
    corruption_rate: float = 0.50
    tokens_per_mask: int = 3
    max_mask_spans: int = 99999
    mask_prefix: str = " [[mask_"
    mask_suffix: str = "]]"
    paradigm_prompt: str = "<|AUTOENCODE + DESHUFFLE|>"
    max_attempts: int = 100
    separator: str = " "
    shuffle: bool = True
    gap_between_spans: int = 1
    paradigm_end: str = ""
    deshuffling_percentage: float = 0.75

    def __post_init__(self):
        """
        Adjust derived fields for min/max corruption or mask sizes.
        """
        self.minimum_corruption_rate = self.corruption_rate * 0.5
        self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
        self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
        self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


class AutoencodingObjective(BaseObjective):
    """
    Autoencoding objective that inserts placeholders for random spans and
    uses the original text up to max_idx as the label.
    """

    def __init__(self, config: AutoencodingConfig) -> None:
        super().__init__(config)

    def build_input_and_labels(
        self, input_text: str, config: AutoencodingConfig
    ) -> Dict[str, Any]:
        """
        Build the 'input_ids' with masked spans, and 'label_ids' with the original tokens.

        If we fail to insert any placeholders after config.max_attempts, we fall back
        to returning unmasked text.
        """
        tokenizer = self.tokenizer
        encoded_input = tokenizer.encode(input_text)

        # Optional paradigm prompt
        prompt_toks = []
        if config.paradigm_prompt:
            prompt_toks = tokenizer.encode(config.paradigm_prompt)

        n_tokens = len(encoded_input)
        max_n_tokens = min(n_tokens, config.remaining_space - 128)
        if max_n_tokens <= 0:
            return {
                "status": "fail",
                "objective": "Autoencoding",
                "input_ids": [],
                "label_ids": [],
                "unused_input_string": input_text,
                "unused_input_ids": encoded_input,
                "masked_count": 0,
                "original_length": 0,
            }

        def sample_span_length(idx):
            raw_len = self.np_rng.poisson(config.tokens_per_mask)
            raw_len = max(raw_len, config.minimum_tokens_per_mask)
            raw_len = min(raw_len, config.maximum_tokens_per_mask)
            limit = max_n_tokens - idx
            if raw_len > limit:
                raw_len = limit
            return max(raw_len, 1)

        # Attempt multiple times to insert at least one masked span
        for attempt_i in range(config.max_attempts):
            input_ids = []
            placeholders_inserted = 0
            tokens_masked = 0
            max_idx = 0
            idx = 0

            input_ids.extend(prompt_toks)

            while idx < max_n_tokens:
                total_so_far = len(input_ids) + max_idx
                if total_so_far >= config.remaining_space:
                    break

                local_corruption_rate = self.np_rng.uniform(
                    config.minimum_corruption_rate, config.maximum_corruption_rate
                )
                span_len = sample_span_length(idx)
                p = local_corruption_rate / span_len

                if self.np_rng.uniform() < p:
                    snippet_end = idx + span_len
                    placeholder_str = f"{config.mask_prefix}{placeholders_inserted}{config.mask_suffix}"
                    ph_toks = tokenizer.encode(placeholder_str)
                    new_in_len = len(input_ids) + len(ph_toks)
                    new_lbl_len = snippet_end
                    prospective_total = new_in_len + new_lbl_len

                    if prospective_total <= config.remaining_space:
                        input_ids.extend(ph_toks)
                        placeholders_inserted += 1
                        tokens_masked += span_len
                        max_idx = max(max_idx, snippet_end)
                        idx += span_len
                        continue

                # else unmasked single token
                prospective_in_len = len(input_ids) + 1
                prospective_lbl_len = max(idx + 1, max_idx)
                prospective_total = prospective_in_len + prospective_lbl_len
                if prospective_total > config.remaining_space:
                    break
                input_ids.append(encoded_input[idx])
                idx += 1
                max_idx = max(max_idx, idx)

            if placeholders_inserted > 0:
                label_ids = encoded_input[:max_idx]
                leftover_ids = encoded_input[max_idx:]
                leftover_str = tokenizer.decode(leftover_ids)

                return {
                    "status": "ok",
                    "objective": "Autoencoding",
                    "input_ids": np.int32(input_ids),
                    "label_ids": np.int32(label_ids),
                    "unused_input_string": leftover_str,
                    "unused_input_ids": np.int32(leftover_ids),
                    "masked_count": tokens_masked,
                    "original_length": max_n_tokens,
                }

        # If no placeholders, fallback to unmasked
        used_ids = encoded_input[:max_n_tokens]
        leftover_ids = encoded_input[max_n_tokens:]
        leftover_str = tokenizer.decode(leftover_ids)
        return {
            "status": "fail",
            "objective": "Autoencoding",
            "input_ids": np.int32(list(prompt_toks) + used_ids.tolist()),
            "label_ids": used_ids,
            "unused_input_string": leftover_str,
            "unused_input_ids": leftover_ids,
            "masked_count": 0,
            "original_length": max_n_tokens,
        }
