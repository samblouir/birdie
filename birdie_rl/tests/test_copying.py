import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.copying import CopyingConfig, CopyingObjective

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_copying_simple(tokenizer):
    config = CopyingConfig(
        paradigm="[COPY_PROMPT] ",
        remaining_space=50
    )
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)

    text = "Copy me, please."
    result = obj(text)
    assert result["status"] == "ok"
    # The label should match the tokens from the used portion
    # except the added paradigm
    used_tokens_count = len(result["label_ids"])
    input_tokens_count = len(result["input_ids"])
    # Paradigm tokens in front => difference in length
    paradigm_toks = obj.tokenizer.encode(config.paradigm)
    assert input_tokens_count == used_tokens_count + len(paradigm_toks)
    decoded_label = tokenizer.decode(result["label_ids"])
    assert "Copy me, please." in decoded_label

def test_copying_empty(tokenizer):
    """Ensure empty input doesn't break it."""
    config = CopyingConfig(remaining_space=10)
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = ""
    out = obj(text)
    assert out["status"] == "ok"
    # label_ids = 0
    assert len(out["label_ids"]) == 0
    # input might have only the paradigm if any
    # or might be empty if no paradigm

def test_copying_unused_text(tokenizer):
    """Check leftover text usage with small remaining space."""
    config = CopyingConfig(remaining_space=2)
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "One two three"
    out = obj(text)
    assert out["status"] == "ok"
    # We only used 2 tokens
    used_decoded = tokenizer.decode(out["input_ids"])
    leftover_str = out["unused_input_string"]
    assert "three" in leftover_str
    assert "three" not in used_decoded

def test_copying_chinese_text(tokenizer):
    """
    Test copying objective with Chinese text to confirm correct copying and leftover text usage.
    """
    config = CopyingConfig(
        paradigm="[COPY_CN] ",
        remaining_space=10
    )
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "你好，世界"
    out = obj(text)
    assert out["status"] == "ok"
    decoded_input = tokenizer.decode(out["input_ids"])
    decoded_label = tokenizer.decode(out["label_ids"])
    assert text in decoded_label, "Label should contain the original Chinese text"
    assert "[COPY_CN]" in decoded_input, "Input must contain the Chinese paradigm prefix"

def test_copying_hindi_text(tokenizer):
    """
    Test copying objective with Hindi text.
    """
    config = CopyingConfig(
        paradigm="[COPY_HI] ",
        remaining_space=6
    )
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया"
    out = obj(text)
    assert out["status"] == "ok"
    decoded_label = tokenizer.decode(out["label_ids"])
    assert text in decoded_label, "Label must contain original Hindi text"

def test_copying_long_dna_sequence(tokenizer):
    """
    Test copying with a lengthy DNA sequence to ensure no break in logic with repeated patterns.
    """
    dna_text = "AGTC" * 30
    config = CopyingConfig(
        paradigm="[DNA_COPY] ",
        remaining_space=60
    )
    obj = CopyingObjective(config)
    obj.set_tokenizer(tokenizer)
    out = obj(dna_text)
    assert out["status"] == "ok"
    decoded_label = tokenizer.decode(out["label_ids"])
    assert dna_text in decoded_label, "Should copy the full DNA sequence if space permits"
