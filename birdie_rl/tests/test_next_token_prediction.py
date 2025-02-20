import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.next_token_prediction import (
    NextTokenPredictionConfig, 
    NextTokenPredictionObjective
)

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_next_token_prediction_basic(tokenizer):
    config = NextTokenPredictionConfig(
        paradigm="(NTP) ",
        remaining_space=30
    )
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "A short text"
    result = obj(text)
    assert result["status"] == "ok"
    # The label is just the raw text tokens
    decoded_input = tokenizer.decode(result["input_ids"])
    decoded_label = tokenizer.decode(result["label_ids"])
    assert "(NTP)" in decoded_input, "Paradigm prefix should appear in the input"
    assert "A short text" in decoded_label, "Label must represent the original text portion"

def test_next_token_prediction_empty(tokenizer):
    """Check behavior with empty input."""
    config = NextTokenPredictionConfig()
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    text = ""
    result = obj(text)
    assert result["status"] == "ok"
    assert len(result["input_ids"]) == 0, "Should yield empty input"
    assert len(result["label_ids"]) == 0, "Should yield empty label"

def test_next_token_prediction_leftover(tokenizer):
    """Ensure leftover tokens appear if remaining_space is small."""
    config = NextTokenPredictionConfig(
        remaining_space=3
    )
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "One two three four five"
    out = obj(text)
    assert out["status"] == "ok"
    leftover_str = out["unused_input_string"]
    used_decoded = tokenizer.decode(out["input_ids"])
    assert "four" in leftover_str or "five" in leftover_str
    assert "four" not in used_decoded or "five" not in used_decoded

def test_next_token_prediction_chinese(tokenizer):
    """
    Test next-token prediction with Chinese text.
    """
    config = NextTokenPredictionConfig(
        paradigm="(NTP-CN) ",
        remaining_space=15
    )
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "你好，世界"
    res = obj(text)
    assert res["status"] == "ok"
    decoded_input = tokenizer.decode(res["input_ids"])
    decoded_label = tokenizer.decode(res["label_ids"])
    assert "(NTP-CN)" in decoded_input
    assert "你好" in decoded_label

def test_next_token_prediction_hindi(tokenizer):
    """
    Test next-token prediction with Hindi text.
    """
    config = NextTokenPredictionConfig(
        paradigm="(NTP-HI) ",
        remaining_space=10
    )
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दोस्तों"
    out = obj(text)
    assert out["status"] == "ok"
    decoded_label = tokenizer.decode(out["label_ids"])
    assert "नमस्ते दोस्तों" in decoded_label

def test_next_token_prediction_dna(tokenizer):
    """
    Test next-token prediction with repeated DNA sequence.
    """
    dna_text = "AGTC" * 5
    config = NextTokenPredictionConfig(
        paradigm="(DNA-PREFIX) ",
        remaining_space=40
    )
    obj = NextTokenPredictionObjective(config)
    obj.set_tokenizer(tokenizer)
    out = obj(dna_text)
    assert out["status"] == "ok"
    decoded_label = tokenizer.decode(out["label_ids"])
    assert dna_text in decoded_label, "All input tokens should appear in label if space permits"
