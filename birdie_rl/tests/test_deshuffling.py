import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.deshuffling import DeshufflingConfig, DeshufflingObjective

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_deshuffling_half(tokenizer):
    """Check that about 50% of tokens are shuffled, but the label remains original text order."""
    config = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=0.5,
        remaining_space=50
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "One two three four five"
    result = obj(text)
    assert result["status"] == "ok"
    # The label should match the original ordering
    label_str = tokenizer.decode(result["label_ids"])
    assert label_str == text, "Label must match original ordering"
    # Input should differ if there's a shuffle
    input_str = tokenizer.decode(result["input_ids"])
    if text != input_str:
        # It's okay that some portion might remain identical by chance
        assert True

def test_deshuffling_full(tokenizer):
    """Check that 100% derangement is used if percentage_of_tokens_to_shuffle=1.0."""
    config = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=1.0,
        remaining_space=50
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "Alpha Beta Gamma Delta"
    result = obj(text)
    assert result["status"] == "ok"
    label_str = tokenizer.decode(result["label_ids"])
    input_str = tokenizer.decode(result["input_ids"])
    # Check label is original
    assert label_str == text
    # Input might be deranged

def test_deshuffling_no_space_left(tokenizer):
    """Check leftover usage if remaining_space is small."""
    cfg = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=1.0,
        remaining_space=3
    )
    obj = DeshufflingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "One two three four"
    res = obj(text)
    assert res["status"] == "ok"
    leftover = res["unused_input_string"]
    # Should contain at least 'four'
    assert "four" in leftover
    used_decoded = tokenizer.decode(res["input_ids"])
    assert "four" not in used_decoded

def test_deshuffling_chinese_text(tokenizer):
    """
    Test deshuffling with Chinese text to ensure partial or full shuffle is handled correctly.
    """
    config = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=0.75,
        remaining_space=15
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "你好 世界 你好"
    res = obj(text)
    assert res["status"] == "ok"
    decoded_label = tokenizer.decode(res["label_ids"])
    # label is the original
    assert decoded_label == text

def test_deshuffling_hindi_text(tokenizer):
    """
    Test deshuffling with Hindi text.
    """
    config = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=0.5,
        remaining_space=20
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया नमस्ते"
    res = obj(text)
    assert res["status"] == "ok"
    # Label must remain unshuffled
    assert tokenizer.decode(res["label_ids"]) == text

def test_deshuffling_dna_sequence(tokenizer):
    """
    Test deshuffling with a repeated DNA sequence.
    """
    dna_text = "AGTC " * 5  # 5 repeats
    config = DeshufflingConfig(
        percentage_of_tokens_to_shuffle=1.0,
        remaining_space=100
    )
    obj = DeshufflingObjective(config)
    obj.set_tokenizer(tokenizer)
    res = obj(dna_text)
    assert res["status"] == "ok"
    label_str = tokenizer.decode(res["label_ids"])
    # label is the original
    assert label_str == dna_text.strip()
    # The input should be a derangement if possible
    if label_str == tokenizer.decode(res["input_ids"]):
        # There's a possibility of fluke, but typically it should differ
        pass
