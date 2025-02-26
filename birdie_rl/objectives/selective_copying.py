"""
SelectiveCopying Objective:
"""

import dataclasses
import numpy as np
from typing import Any, Dict
from birdie_rl.objectives.base import BaseObjective, BaseObjectiveConfig

# Some tokens we inject (unchanged)
COPY_TOKEN      = "[COPY]"
START_TOKEN     = "find"
RESULT_TOKEN    = "result"
END_TOKEN       = "/find"
CONTEXT_TOKEN   = "\n\n[context]\n"
CLOSE_CONTEXT   = "\n[/context]"
SEP_TOKEN       = "sep"
DONE_TOKEN      = "\n\n[done]"


@dataclasses.dataclass
class SelectiveCopyingConfig(BaseObjectiveConfig):
    corruption_rate: float = 0.5
    tokens_per_mask: int = 8
    shuffle: bool = True
    separator: str = " "
    paradigm_prompt: str = "<|Selective Copying|>"
    gap_between_spans: int = 1
    max_attempts: int = 100
    paradigm_end: str = ""
    min_delimiter_prefix_length: int = 8
    max_delimiter_prefix_length: int = 64
    min_delimiter_suffix_length: int = 8
    max_delimiter_suffix_length: int = 64
    format_style: str = "query_context"

    def __post_init__(self):
        """
        Adjusts some derived fields for min/max corruption rates
        and tokens_per_mask range. 
        """
        self.minimum_corruption_rate = self.corruption_rate * 0.5
        self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
        self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
        self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)


class SelectiveCopyingObjective(BaseObjective):
    """
    SelectiveCopyingObjective (Optimized).

    Overview (unchanged in logic):
      1) Takes text + optional paradigm_prompt => merges them in `input_ids`.
      2) Iterates tokens, deciding whether to mask a span or pass a single token unmasked.
      3) Each masked span => we place the placeholder in input_ids,
         and we append [placeholder + masked_tokens] to label_ids.
      4) If we insert >= 1 placeholder, we finalize. Otherwise, we try again up to max_attempts.
      5) If still no placeholders => return unmasked version.
    """

    def __init__(self, config: SelectiveCopyingConfig) -> None:
        super().__init__(config)

    def build_input_and_labels(
        self, input_text: str, config: SelectiveCopyingConfig
    ) -> Dict[str, Any]:
        """
        Build "input_ids" (with placeholders) and "label_ids" ([placeholder + tokens])
        according to SelectiveCopying logic.

        Args:
            input_text: Raw string to be infilled.
            config: SelectiveCopyingConfig specifying corruption rates, tokens_per_mask, etc.

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

        tokenizer = self.tokenizer
        encode = tokenizer.encode
        decode = tokenizer.decode

        # Pre-encode repeated strings
        enc_context_token = encode(CONTEXT_TOKEN)
        enc_close_context = encode(CLOSE_CONTEXT)
        enc_done = encode(DONE_TOKEN)

        # Encode input and prompt
        encoded_input = encode(input_text)
        prompt_toks = encode(config.paradigm_prompt) if config.paradigm_prompt else []

        n_tokens = len(encoded_input)
        # We'll reserve a bit of space for placeholders. 
        # Original code used: config.remaining_space - 128 as max_n_tokens,
        # so we keep that:
        max_n_tokens = config.remaining_space - 128
        if max_n_tokens <= 0:
            return {
                "status": "fail",
                "objective": "SelectiveCopying",
                "input_ids": [],
                "label_ids": [],
                "unused_input_string": input_text,
                "unused_input_ids": encoded_input,
                "masked_count": 0,
                "original_length": 0,
            }

        # Local references for random draws
        np_rng = self.np_rng
        min_sp = config.minimum_tokens_per_mask
        max_sp = config.maximum_tokens_per_mask

        def sample_span_length(start_idx: int) -> int:
            """
            Poisson-based sampling for number of tokens in a masked span, 
            clamped by config and ensuring we don't exceed max_n_tokens 
            from the current index.
            """
            raw_len = np_rng.poisson(config.tokens_per_mask)
            raw_len = max(raw_len, min_sp)
            raw_len = min(raw_len, max_sp)
            limit = max_n_tokens - start_idx
            if raw_len > limit:
                raw_len = limit
            return max(raw_len, 1)

        # Prepare a base input that always has prompt + context tokens in place
        # (We won't re-deepcopy on each attempt; we just re-build).
        # The actual final "input_ids" will be formed after placeholders are determined.
        base_input_ids = [
            *prompt_toks,
            *enc_context_token,
            *enc_close_context,
        ]

        # Attempt multiple times to insert placeholders
        for attempt_i in range(config.max_attempts):
            label_blocks = []
            placeholders_inserted = 0
            masked_tokens_count = 0

            # We'll build the context tokens and placeholder instructions as we go:
            built_context = []
            built_instructions_ctr = 0
            # We'll store unshuffled_instructions as (placeholder_idx, prefix, snippet, suffix)
            unshuffled_instructions = []

            initial_delimiter_prefix_length = np_rng.integers(
                config.min_delimiter_prefix_length, config.max_delimiter_prefix_length
            )

            idx = initial_delimiter_prefix_length
            # sum of label blocks so far
            lbl_len_current = 0
            while idx < n_tokens:
                in_len = len(base_input_ids) + built_instructions_ctr + idx
                # label blocks so far plus eventual DONE token
                lbl_len = lbl_len_current + len(enc_done)
                total_so_far = in_len + lbl_len

                if total_so_far >= config.remaining_space:
                    # If it's the last attempt and no placeholders inserted,
                    # just break and we'll exit with unmasked result.
                    break

                # local corruption rate
                local_corruption_rate = np_rng.uniform(
                    config.minimum_corruption_rate, config.maximum_corruption_rate
                )
                span_len = sample_span_length(idx)
                # Probability that we mask
                p = local_corruption_rate / span_len

                # Attempt to mask with probability p
                if np_rng.uniform() < p:
                    snippet = encoded_input[idx : idx + span_len]
                    if len(snippet) > 0:
                        # random prefix / suffix lengths
                        initial_delimiter_suffix_length = np_rng.integers(
                            config.min_delimiter_suffix_length,
                            config.max_delimiter_suffix_length,
                        )
                        prefix_delimiter = encoded_input[
                            idx - initial_delimiter_prefix_length : idx
                        ]
                        suffix_delimiter = encoded_input[
                            idx + span_len : idx + span_len + initial_delimiter_suffix_length
                        ]

                        initial_delimiter_prefix_length = np_rng.integers(
                            config.min_delimiter_prefix_length,
                            config.max_delimiter_prefix_length,
                        )

                        # We always format placeholders and labels the same way:
                        # label = f"\n\n[{RESULT_TOKEN} X]\n" + snippet
                        # input placeholder = f"\n\n[{START_TOKEN} X]\n" + prefix + ...
                        # But we must check prospective space usage.
                        # Build the would-be encoded tokens for label
                        if placeholders_inserted == 0:
                            lbl_str = f"[{RESULT_TOKEN} {placeholders_inserted}]\n"
                        else:
                            lbl_str = f"\n\n[{RESULT_TOKEN} {placeholders_inserted}]\n"

                        lb_toks = [*encode(lbl_str), *snippet]
                        ph_toks = [
                            *encode(f"\n\n[{START_TOKEN} {placeholders_inserted}]\n"),
                            *prefix_delimiter,
                            *encode(f"\n[{SEP_TOKEN}]\n"),
                            *suffix_delimiter,
                            *encode(f"\n[/{START_TOKEN} {placeholders_inserted}]"),
                        ]

                        prospective_in_len = in_len + len(ph_toks)
                        prospective_lbl_len = lbl_len_current + len(lb_toks)
                        prospective_total = prospective_in_len + prospective_lbl_len + len(enc_done)

                        if prospective_total <= config.remaining_space:
                            # Accept
                            built_instructions_ctr += len(ph_toks)
                            unshuffled_instructions.append(
                                (placeholders_inserted, prefix_delimiter, snippet, suffix_delimiter)
                            )
                            label_blocks.append(lb_toks)
                            lbl_len_current += len(lb_toks)
                            placeholders_inserted += 1
                            masked_tokens_count += span_len
                            idx += span_len
                            continue
                    # If snippet empty or doesn't fit, fall back to unmasked below

                # If not masking => add a single unmasked token
                prospective_in_len = len(base_input_ids) + built_instructions_ctr + idx + 1
                prospective_total = prospective_in_len + lbl_len_current + len(enc_done)
                if prospective_total > config.remaining_space:
                    break

                built_context.append(encoded_input[idx])
                idx += 1

            # If we inserted placeholders, finalize
            if placeholders_inserted > 0:
                # Possibly shuffle
                if config.shuffle:
                    shuffle_indices = np_rng.permutation(len(unshuffled_instructions))
                    unshuffled_instructions = [unshuffled_instructions[i] for i in shuffle_indices]

                    built_instructions = []
                    label_ids = []
                    for placeholder_idx, (orig_pl_idx, prefix_delimiter, snippet, suffix_delimiter) in enumerate(unshuffled_instructions):
                        if placeholder_idx == 0:
                            lbl_str = f"[{RESULT_TOKEN} {placeholder_idx}]\n"
                        else:
                            lbl_str = f"\n\n[{RESULT_TOKEN} {placeholder_idx}]\n"
                        lb_toks = [*encode(lbl_str), *snippet]
                        label_ids.extend(lb_toks)

                        ph_toks = [
                            *encode(f"\n\n[{START_TOKEN} {placeholder_idx}]\n"),
                            *prefix_delimiter,
                            *encode(f"\n[{SEP_TOKEN}]\n"),
                            *suffix_delimiter,
                            *encode(f"\n[/{START_TOKEN} {placeholder_idx}]"),
                        ]
                        built_instructions.extend(ph_toks)

                    label_ids.extend(enc_done)

                else:
                    # If no shuffle, keep label blocks in order
                    label_ids = []
                    for block in label_blocks:
                        label_ids.extend(block)
                    label_ids.extend(enc_done)

                    # Then shuffle instructions physically but keep the same content.
                    shuffle_indices = np_rng.permutation(len(unshuffled_instructions))
                    unshuffled_instructions = [unshuffled_instructions[i] for i in shuffle_indices]
                    built_instructions = []
                    for placeholder_idx, (orig_pl_idx, prefix_delimiter, snippet, suffix_delimiter) in enumerate(unshuffled_instructions):
                        # The actual placeholder text doesn't rely on the shuffle index for meaning,
                        # but the original code used "placeholder_idx" to label it. We keep that.
                        ph_toks = [
                            *encode(f"\n\n[{START_TOKEN} {placeholder_idx}]\n"),
                            *prefix_delimiter,
                            *encode(f"\n[{SEP_TOKEN}]\n"),
                            *suffix_delimiter,
                            *encode(f"\n[/{START_TOKEN} {placeholder_idx}]"),
                        ]
                        built_instructions.extend(ph_toks)

                # Build final input
                context_block = [
                    *enc_context_token,
                    *encoded_input[:idx],
                    *enc_close_context,
                ]
                if config.format_style == "query_context":
                    input_ids = [
                        *prompt_toks,
                        *built_instructions,
                        *context_block,
                    ]
                else:
                    input_ids = [
                        *prompt_toks,
                        *context_block,
                        *built_instructions,
                    ]

                leftover_ids = encoded_input[idx:]
                leftover_text = decode(leftover_ids) if len(leftover_ids) else ""
                return {
                    "status": "ok",
                    "objective": "SelectiveCopying",
                    "input_ids": input_ids,
                    "label_ids": label_ids,
                    "unused_input_string": leftover_text,
                    "unused_input_ids": leftover_ids,
                    "masked_count": masked_tokens_count,
                    "original_length": max_n_tokens,
                }

        # If we get here => No placeholders after all attempts => return unmasked text
        used_ids = encoded_input[:max_n_tokens]
        leftover_ids = encoded_input[max_n_tokens:]
        leftover_text = decode(leftover_ids) if len(leftover_ids) else ""
        return {
            "status": "ok",
            "objective": "SelectiveCopying",
            "input_ids": used_ids.tolist(),
            "label_ids": [],
            "unused_input_string": leftover_text,
            "unused_input_ids": leftover_ids,
            "masked_count": 0,
            "original_length": max_n_tokens,
        }


# -------------------- Small Test for Output Matching --------------------
if __name__ == "__main__":
    # We will copy the original class into this file under a different name
    # so we can do a side-by-side test that the outputs match.

    # (You would normally import the original from your existing file;
    # here, we replicate it briefly for demonstration.)

    @dataclasses.dataclass
    class SelectiveCopyingConfigOriginal(BaseObjectiveConfig):
        corruption_rate: float = 0.5
        tokens_per_mask: int = 8
        shuffle: bool = True
        separator: str = " "
        paradigm_prompt: str = "<|Selective Copying|>"
        gap_between_spans: int = 1
        max_attempts: int = 100
        paradigm_end: str = ""
        min_delimiter_prefix_length: int = 2
        max_delimiter_prefix_length: int = 12
        min_delimiter_suffix_length: int = 2
        max_delimiter_suffix_length: int = 12
        format_style: str = "query_context"

        def __post_init__(self):
            self.minimum_corruption_rate = self.corruption_rate * 0.5
            self.maximum_corruption_rate = min(0.95, self.corruption_rate * 2.0)
            self.minimum_tokens_per_mask = max(1, self.tokens_per_mask // 3)
            self.maximum_tokens_per_mask = max(1, self.tokens_per_mask * 3)

    class SelectiveCopyingObjectiveOriginal(BaseObjective):
        def __init__(self, config: SelectiveCopyingConfigOriginal) -> None:
            super().__init__(config)

        def build_input_and_labels(
            self, input_text: str, config: SelectiveCopyingConfigOriginal
        ) -> Dict[str, Any]:
            # -- The original code verbatim (pasted from the question) --
            tokenizer = self.tokenizer
            encoded_input = tokenizer.encode(input_text)

            prompt_toks = []
            if config.paradigm_prompt:
                prompt_toks = tokenizer.encode(config.paradigm_prompt)

            n_tokens = len(encoded_input)
            max_n_tokens = config.remaining_space - 128
            if max_n_tokens <= 0:
                return {
                    "status": "fail",
                    "objective": "SelectiveCopying",
                    "input_ids": [],
                    "label_ids": [],
                    "unused_input_string": input_text,
                    "unused_input_ids": encoded_input,
                    "masked_count": 0,
                    "original_length": 0,
                }

            def sample_span_length(start_idx: int) -> int:
                raw_len = self.np_rng.poisson(config.tokens_per_mask)
                raw_len = max(raw_len, config.minimum_tokens_per_mask)
                raw_len = min(raw_len, config.maximum_tokens_per_mask)
                limit = max_n_tokens - start_idx
                if raw_len > limit:
                    raw_len = limit
                return max(raw_len, 1)

            CONTEXT_TOKEN = "\n\n[context]\n"
            CLOSE_CONTEXT = "\n[/context]"
            DONE_TOKEN = "\n\n[done]"

            import copy
            input_ids = [
                *prompt_toks,
                *self.tokenizer.encode(CONTEXT_TOKEN),
                *self.tokenizer.encode(CLOSE_CONTEXT),
            ]

            encoded_done = tokenizer.encode(DONE_TOKEN)
            label_blocks = []
            placeholders_inserted = 0
            masked_tokens_count = 0

            # We'll attempt multiple times:
            for attempt_i in range(config.max_attempts):
                label_blocks = []
                placeholders_inserted = 0
                masked_tokens_count = 0
                built_instructions_ctr = 0
                built_context = []
                unshuffled_instructions = []

                initial_delimiter_prefix_length = self.np_rng.integers(
                    config.min_delimiter_prefix_length, config.max_delimiter_prefix_length
                )
                idx = initial_delimiter_prefix_length
                while idx < n_tokens:
                    in_len = len(input_ids) + built_instructions_ctr + idx
                    lbl_len = sum(len(b) for b in label_blocks) + len(encoded_done)
                    total_so_far = in_len + lbl_len
                    if total_so_far >= config.remaining_space:
                        break

                    local_corruption_rate = self.np_rng.uniform(
                        config.minimum_corruption_rate,
                        config.maximum_corruption_rate
                    )
                    span_len = sample_span_length(idx)
                    p = local_corruption_rate / span_len

                    if self.np_rng.uniform() < p:
                        snippet = encoded_input[idx : idx + span_len]
                        initial_delimiter_suffix_length = self.np_rng.integers(
                            config.min_delimiter_suffix_length,
                            config.max_delimiter_suffix_length,
                        )
                        prefix_delimiter = encoded_input[
                            idx - initial_delimiter_prefix_length : idx
                        ]
                        suffix_delimiter = encoded_input[
                            idx + span_len : idx + span_len + initial_delimiter_suffix_length
                        ]
                        initial_delimiter_prefix_length = self.np_rng.integers(
                            config.min_delimiter_prefix_length,
                            config.max_delimiter_prefix_length,
                        )
                        if len(snippet) > 0:
                            if placeholders_inserted == 0:
                                lbl_str = f"[result {placeholders_inserted}]\n"
                            else:
                                lbl_str = f"\n\n[result {placeholders_inserted}]\n"
                            lb_toks = [*tokenizer.encode(lbl_str), *snippet]

                            ph_toks = [
                                *tokenizer.encode(f"\n\n[find {placeholders_inserted}]\n"),
                                *prefix_delimiter,
                                *tokenizer.encode(f"\n[sep]\n"),
                                *suffix_delimiter,
                                *tokenizer.encode(f"\n[/find {placeholders_inserted}]"),
                            ]
                            prospective_in_len  = in_len + len(ph_toks)
                            prospective_lbl_len = sum(len(b) for b in label_blocks) + len(lb_toks)
                            prospective_total   = prospective_in_len + prospective_lbl_len + len(encoded_done)

                            if prospective_total <= config.remaining_space:
                                built_instructions_ctr += len(ph_toks)
                                unshuffled_instructions.append(
                                    (placeholders_inserted, prefix_delimiter, snippet, suffix_delimiter)
                                )
                                label_blocks.append(lb_toks)
                                placeholders_inserted += 1
                                masked_tokens_count += span_len
                                idx += span_len
                                continue

                    prospective_in_len = len(input_ids) + built_instructions_ctr + idx + 1
                    prospective_total  = prospective_in_len + sum(len(b) for b in label_blocks) + len(encoded_done)
                    if prospective_total > config.remaining_space:
                        break

                    built_context.append(encoded_input[idx])
                    idx += 1

                if placeholders_inserted > 0:
                    if config.shuffle:
                        shuffle_indices = self.np_rng.permutation(len(unshuffled_instructions))
                        unshuffled_instructions = [unshuffled_instructions[i] for i in shuffle_indices]

                        built_instructions = []
                        label_ids = []
                        for placeholder_idx, (orig_pl_idx, prefix_delimiter, snippet, suffix_delimiter) in enumerate(unshuffled_instructions):
                            if placeholder_idx == 0:
                                lbl_str = f"[result {placeholder_idx}]\n"
                            else:
                                lbl_str = f"\n\n[result {placeholder_idx}]\n"
                            lb_toks = [*tokenizer.encode(lbl_str), *snippet]
                            label_ids.extend(lb_toks)

                            ph_toks = [
                                *tokenizer.encode(f"\n\n[find {placeholder_idx}]\n"),
                                *prefix_delimiter,
                                *tokenizer.encode(f"\n[sep]\n"),
                                *suffix_delimiter,
                                *tokenizer.encode(f"\n[/find {placeholder_idx}]"),
                            ]
                            built_instructions.extend(ph_toks)
                        label_ids.extend(encoded_done)
                    else:
                        label_ids = []
                        for block in label_blocks:
                            label_ids.extend(block)
                        label_ids.extend(encoded_done)

                        shuffle_indices = self.np_rng.permutation(len(unshuffled_instructions))
                        unshuffled_instructions = [unshuffled_instructions[i] for i in shuffle_indices]
                        built_instructions = []
                        for placeholder_idx, (orig_pl_idx, prefix_delimiter, snippet, suffix_delimiter) in enumerate(unshuffled_instructions):
                            ph_toks = [
                                *tokenizer.encode(f"\n\n[find {placeholder_idx}]\n"),
                                *prefix_delimiter,
                                *tokenizer.encode(f"\n[sep]\n"),
                                *suffix_delimiter,
                                *tokenizer.encode(f"\n[/find {placeholder_idx}]"),
                            ]
                            built_instructions.extend(ph_toks)

                    CONTEXT_TOKEN = "\n\n[context]\n"
                    CLOSE_CONTEXT = "\n[/context]"
                    context_block = [
                        *tokenizer.encode(CONTEXT_TOKEN),
                        *encoded_input[:idx],
                        *tokenizer.encode(CLOSE_CONTEXT),
                    ]
                    if config.format_style == "query_context":
                        input_ids = [
                            *prompt_toks,
                            *built_instructions,
                            *context_block,
                        ]
                    else:
                        input_ids = [
                            *prompt_toks,
                            *context_block,
                            *built_instructions,
                        ]

                    leftover_ids = encoded_input[idx:]
                    leftover_text = tokenizer.decode(leftover_ids)
                    return {
                        "status": "ok",
                        "objective": "SelectiveCopying",
                        "input_ids": input_ids,
                        "label_ids": label_ids,
                        "unused_input_string": leftover_text if len(leftover_text) else "",
                        "unused_input_ids": leftover_ids,
                        "masked_count": masked_tokens_count,
                        "original_length": max_n_tokens,
                    }

            used_ids = encoded_input[:max_n_tokens]
            leftover_ids = encoded_input[max_n_tokens:]
            leftover_text = tokenizer.decode(leftover_ids)
            return {
                "status": "ok",
                "objective": "SelectiveCopying",
                "input_ids": used_ids.tolist(),
                "label_ids": [],
                "unused_input_string": leftover_text,
                "unused_input_ids": leftover_ids,
                "masked_count": 0,
                "original_length": max_n_tokens,
            }

    # --------------------------------------------------------------------
    # Actual test: We'll set a fixed seed, run both objective classes,
    # and confirm the outputs match (field by field).
    # --------------------------------------------------------------------
    from birdie_rl.modeling.tokenizer import Tokenizer

    # Create same config for both
    config_original = SelectiveCopyingConfigOriginal(
        remaining_space=90,  # enough space for a small test
        corruption_rate=0.3,
        tokens_per_mask=2,
        shuffle=True,
        format_style="query_context",
        max_attempts=5,
    )
    config_optimized = SelectiveCopyingConfig(
        remaining_space=90,
        corruption_rate=0.3,
        tokens_per_mask=2,
        shuffle=True,
        format_style="query_context",
        max_attempts=5,
    )

    text_to_test = "Hello world. This is a short test for selective copying."

    # Create both objectives
    obj_original = SelectiveCopyingObjectiveOriginal(config_original)
    obj_optimized = SelectiveCopyingObjective(config_optimized)

    # Use the same tokenizer and the same random seed
    tok = Tokenizer()
    obj_original.set_tokenizer(tok)
    obj_optimized.set_tokenizer(tok)

    # Make sure they both start with the same random state:
    # (Numpy RNG is inside each class, so we set each one identically)
    seed_val = 123
    obj_original.np_rng = np.random.default_rng(seed_val)
    obj_optimized.np_rng = np.random.default_rng(seed_val)

    out_original = obj_original.build_input_and_labels(text_to_test, config_original)
    out_optimized = obj_optimized.build_input_and_labels(text_to_test, config_optimized)

    # Compare results
    fields = ["status", "objective", "input_ids", "label_ids", "unused_input_string",
              "unused_input_ids", "masked_count", "original_length"]

    # Print a small summary
    print("===== COMPARISON TEST =====")
    all_match = True
    for f in fields:
        v_orig = out_original[f]
        v_opt = out_optimized[f]
        if v_orig != v_opt:
            all_match = False
            print(f"Mismatch in field '{f}':")
            print("  Original:", v_orig)
            print("  Optimized:", v_opt)
            print()
        else:
            print(f"Match in field '{f}' ✓")

    if all_match:
        print("\nAll fields match perfectly!")
    else:
        print("\nThere were mismatches! Investigate the differences above.")
