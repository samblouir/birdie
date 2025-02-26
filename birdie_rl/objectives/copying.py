"""
Copying objective
"""

import dataclasses
import numpy as np
from typing import Any, Dict

from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig
from birdie_rl.objectives.utils import slice_text_by_remaining_space


@dataclasses.dataclass
class CopyingConfig(BaseObjectiveConfig):
    """
    Configuration for the CopyingObjective.

    Attributes:
        paradigm: Optional string to prepend to the input tokens.
        special_token_ids: Dictionary for any special tokens (unused).
    """
    paradigm: str = "<|COPY|>"
    paradigm_suffix: str = ""
    special_token_ids: dict = dataclasses.field(default_factory=dict)


class CopyingObjective(BaseObjective):
    """
    Copying objective (no teacher forcing):
      - The label is exactly the used portion of the text.
      - The input is [paradigm] + used_tokens.
      - If (len(input_ids) + len(label_ids)) > remaining_space => skip the sample.
    """

    def __init__(self, config: CopyingConfig) -> None:
        super().__init__(config)

    def build_input_and_labels(self, input_text: str, config: CopyingConfig) -> Dict[str, Any]:
        """
        Build the dictionary with "input_ids" and "label_ids" for the copying objective
        without partial truncation. If the total would exceed `remaining_space`, we skip.
        """
        
        if config.paradigm:
            p_toks = self.tokenizer.encode(config.paradigm)
            p_toks = self.safe_cast_to_list(p_toks)
            
        # Slice up to config.remaining_space (no dividing by 2).
        slice_data = slice_text_by_remaining_space(
            text=input_text,
            tokenizer=self.tokenizer,
            remaining_space=config.remaining_space//2 - len(p_toks),
        )
        used_tokens = slice_data["used_tokens"]     # np.array of used tokens
        leftover_text = slice_data["unused_text"]   # leftover raw text
        leftover_tokens = slice_data["unused_tokens"]

        # Build input_ids = [paradigm tokens] + used_tokens
        final_input_tokens = []
        if config.paradigm:
            final_input_tokens.extend(p_toks)
        final_input_tokens.extend(self.safe_cast_to_list(used_tokens))
        input_ids = np.array(final_input_tokens, dtype=np.int32)

        # Label is exactly the used tokens
        label_ids = used_tokens  # already np.array

        # Check if (input_ids + label_ids) fits in remaining_space
        total_length = len(input_ids) + len(label_ids)
        if total_length > config.remaining_space:
            # The sample doesn't fit -> skip
            return {
                "status": "error",
                "message": (
                    f"Copying - Sample too large. Needed {total_length} tokens, "
                    f"but only have {config.remaining_space}."
                ),
                "objective": "Copying",
            }

        # If it fits, return "ok"
        return {
            "status": "ok",
            "objective": "Copying",
            "input_ids": input_ids,
            "label_ids": label_ids,
            "unused_input_string": leftover_text,
            "unused_input_ids": leftover_tokens,
        }


# -----------------------------------------
# Minimal test for the CopyingObjective
# -----------------------------------------
if __name__ == "__main__":
    from modeling.tokenizer import Tokenizer

    text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
    config = CopyingConfig(
        remaining_space=15,  # e.g. small to demonstrate skipping or leftover
        paradigm="[COPY_PROMPT] "
    )
    obj = CopyingObjective(config)
    obj.set_tokenizer(Tokenizer())
    result = obj(text)
    
    print("\n=== Copying Objective Test ===")
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
