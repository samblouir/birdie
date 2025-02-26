"""
Deshuffling objective
"""

import dataclasses
import numpy as np
from typing import Any, Dict

from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig
from birdie_rl.objectives.utils import slice_text_by_remaining_space


@dataclasses.dataclass
class DeshufflingConfig(BaseObjectiveConfig):
    """
    Configuration for the DeshufflingObjective.

    Attributes:
        paradigm: Optional string to prepend to the input tokens.
        special_token_ids: Dictionary for any special tokens (unused).
    """
    paradigm: str = "<|DESHUFFLE|>"
    paradigm_suffix: str = ""
    special_token_ids: dict = dataclasses.field(default_factory=dict)
    percentage_to_shuffle: float = 0.75


class DeshufflingObjective(BaseObjective):
    """
    Deshuffling objective:
      - The label is exactly the used portion of the text.
      - The input is [paradigm] + used_tokens.
      - If (len(input_ids) + len(label_ids)) > remaining_space => skip the sample.
    """

    def __init__(self, config: DeshufflingConfig) -> None:
        super().__init__(config)

    def build_input_and_labels(self, input_text: str, config: DeshufflingConfig) -> Dict[str, Any]:
        """
        Build the dictionary with "input_ids" and "label_ids" for the Deshuffling objective.
        This version skips samples that would exceed `remaining_space`.
        """
        # Tokenize the paradigm if provided
        p_toks = self.tokenizer.encode(config.paradigm) if config.paradigm else np.array([], dtype=np.int32)
        
        # Determine available space for the used tokens
        available_space = int(((config.remaining_space // 2)*0.90 - len(p_toks)))
        slice_data = slice_text_by_remaining_space(
            text=input_text,
            tokenizer=self.tokenizer,
            remaining_space=available_space,
        )
        used_tokens = slice_data["used_tokens"]     # np.array of used tokens
        leftover_text = slice_data["unused_text"]     # leftover raw text
        leftover_tokens = slice_data["unused_tokens"]

        # Start building the input with paradigm tokens (if any)
        final_input_tokens = p_toks

        # Decode the used tokens to a string
        decoded_used_tokens = self.tokenizer.decode(used_tokens)

        # Determine if we should split into words or characters.
        # If there are more than 2 spaces, treat the tokens as words.
        # has_spaces = 2 < decoded_used_tokens.count(' ')
        has_spaces = False
        tokens = decoded_used_tokens.split(' ') if has_spaces else list(decoded_used_tokens)

        # Convert tokens to a NumPy array for vectorized assignment.
        tokens_arr = np.array(tokens)
        n_tokens = len(tokens_arr)
        n_shuffle = int(n_tokens * config.percentage_to_shuffle)

        if n_shuffle > 0:
            # Choose indices and perform vectorized shuffling.
            indices_to_shuffle = self.np_rng.choice(n_tokens, n_shuffle, replace=False)
            tokens_arr[indices_to_shuffle] = self.np_rng.permutation(tokens_arr[indices_to_shuffle])

        # Reassemble tokens into a string.
        final_string = ' '.join(tokens_arr.tolist()) if has_spaces else ''.join(tokens_arr.tolist())
        shuffled_used_tokens = self.tokenizer.encode(final_string)

        # Concatenate the paradigm tokens and the shuffled used tokens.
        final_input_tokens = np.concatenate((final_input_tokens, shuffled_used_tokens))
        input_ids = np.array(final_input_tokens, dtype=np.int32)

        # The label is the original used tokens.
        label_ids = used_tokens

        # Check if the overall sample fits in the remaining space.
        total_length = len(input_ids) + len(label_ids)
        if total_length > config.remaining_space:
            return {
                "status": "error",
                "message": (
                    f"Deshuffling - Sample too large. Needed {total_length} tokens, "
                    f"but only have {config.remaining_space}."
                ),
                "objective": "Deshuffling",
            }

        return {
            "status": "ok",
            "objective": "Deshuffling",
            "input_ids": input_ids,
            "label_ids": label_ids,
            "unused_input_string": leftover_text,
            "unused_input_ids": leftover_tokens,
        }

# -----------------------------------------
# Minimal test for the DeshufflingObjective
# -----------------------------------------
if __name__ == "__main__":
    # from modeling.tokenizer import Tokenizer

    from birdie_rl.modeling.tokenizer import Tokenizer

    text = "abcdefghijklmnopqrstuvwxyz"*64
    config = DeshufflingConfig(
        remaining_space=1024,  # e.g. small to demonstrate skipping or leftover
        # paradigm="[COPY_PROMPT] "
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(Tokenizer())
    result = obj(text)
    
    print("\n=== Deshuffling Objective Test ===")
    print("Status:", result["status"])
    if result["status"] == "ok":
        print("Input IDs:", result["input_ids"])
        print("Decoded Input:", obj.tokenizer.decode(result["input_ids"]))
        print("Label IDs:", result["label_ids"])
        print("Decoded Label:", obj.tokenizer.decode(result["label_ids"]))
        print("Unused text:", result["unused_input_string"])
        print("Unused tokens:", result["unused_input_ids"])
    else:
        # If it doesn't fit
        print("Error message:", result.get("message", "No message"))
