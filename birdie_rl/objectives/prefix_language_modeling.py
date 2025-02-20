"""
Prefix Language Modeling objective.

The input is the first X% of tokens (prefix), the label is the remaining tokens (suffix).

Changes:
- Accepts a 'paradigm_str' in the config.
- No longer uses `truncate_for_teacher_forcing`.
- Uses `slice_text_by_remaining_space` to get the token slice.
- Skips the sample if (prefix + suffix + paradigm) would exceed `remaining_space`.
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
        special_token_ids: Optional dict of special tokens (unused in this example).
    """
    prefix_fraction: float = 0.75
    paradigm_str: str = "<|PREFIX LM|>"
    special_token_ids: dict = dataclasses.field(default_factory=dict)


class PrefixLanguageModelingObjective(BaseObjective):
    """
    For prefix LM: 
      - The input is [paradigm_str] + the first `prefix_fraction` of tokens.
      - The label is the remaining tokens.

    No post-hoc truncation is performed. If the combined length (input + label) 
    would exceed `remaining_space`, the sample is skipped.
    """

    def __init__(self, config: PrefixLanguageModelingConfig) -> None:
        super().__init__(config)

    def build_input_and_labels(
        self, input_text: str, config: PrefixLanguageModelingConfig
    ) -> Dict[str, Any]:
        """
        Construct the prefix as input and the suffix as label.
        No teacher-forcing style truncation is used.
        """
        paradigm_tokens = self.tokenizer.encode(config.paradigm_str)
        
        

        # 1) Slice the text up to config.remaining_space (no partial usage beyond that).
        slice_data = slice_text_by_remaining_space(
            text=input_text,
            tokenizer=self.tokenizer,
            remaining_space=(config.remaining_space - len(paradigm_tokens)),
        )
        used_tokens = slice_data["used_tokens"]         # up to remaining_space
        leftover_text = slice_data["unused_text"]       # leftover raw text
        leftover_tokens = slice_data["unused_tokens"]   # leftover tokens

        used_tokens = np.array(used_tokens, dtype=np.int32)
        length = len(used_tokens)

        # 2) If too short, just return the entire used_tokens for both input & label 
        #    (or you can skip; but here we do the fallback).
        if length < 2:
            return {
                "status": "ok",
                "objective": "Prefix Language Modeling",
                # Input = used tokens
                "input_ids": used_tokens,
                # Label = the same (not ideal, but fallback if there's nothing else)
                "label_ids": used_tokens,
                "unused_input_string": leftover_text,
                "unused_input_ids": leftover_tokens,
            }

        # 3) Compute prefix length and split into prefix/suffix
        prefix_len = int(np.floor(length * config.prefix_fraction))
        prefix_len = max(0, min(prefix_len, length))
        prefix = used_tokens[:prefix_len]
        suffix = used_tokens[prefix_len:]

        # 4) Prepend the paradigm_str tokens to the prefix
        final_input_ids = []
        if config.paradigm_str:
            final_input_ids.extend(self.safe_cast_to_list(paradigm_tokens))
        final_input_ids.extend(self.safe_cast_to_list(prefix))

        final_input_ids = np.array(final_input_ids, dtype=np.int32)
        final_label_ids = suffix

        # 5) Check total length
        total_len = len(final_input_ids) + len(final_label_ids)
        if total_len > config.remaining_space:
            # Skip this sample if it doesn't fit
            return {
                "status": "error",
                "message": (
                    f"Prefix LM - Combined length {total_len} exceeds "
                    f"remaining_space={config.remaining_space}. Skipping."
                ),
                "objective": "Prefix Language Modeling",
            }

        # 6) Return result
        return {
            "status": "ok",
            "objective": "Prefix Language Modeling",
            "input_ids": final_input_ids,
            "label_ids": final_label_ids,
            "unused_input_string": leftover_text,
            "unused_input_ids": leftover_tokens,
        }


# Example demo
if __name__ == "__main__":
    from modeling.tokenizer import Tokenizer

    tok = Tokenizer()
    text = "This is a test for prefix language modeling. " * 3

    cfg = PrefixLanguageModelingConfig(
        remaining_space=40,
        prefix_fraction=0.7,
        paradigm_str="<<PREFIX>> "
    )
    obj = PrefixLanguageModelingObjective(cfg)
    obj.set_tokenizer(tok)
    result = obj(text)

    print("\n=== Prefix LM Demo ===")
    print("Status:", result["status"])
    if result["status"] == "ok":
        print("Original text (short):", text[:60], "...")
        print("Input IDs:", result["input_ids"])
        print("Decoded Input:", tok.decode(result["input_ids"]))
        print("Label IDs:", result["label_ids"])
        print("Decoded Label:", tok.decode(result["label_ids"]))
        print("Unused text:", result["unused_input_string"])
    else:
        print("Error message:", result.get("message", "No message"))
