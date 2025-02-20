import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.selective_copying import (
    SelectiveCopyingConfig,
    SelectiveCopyingObjective,
    CONTEXT_TOKEN,
    CLOSE_CONTEXT_TOKEN
)

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_selective_copying_query_context(tokenizer):
    """
    For 'query_context', the instructions should appear first, then [CONTEXT] <original> [/CONTEXT].
    """
    cfg = SelectiveCopyingConfig(
        formatting_type="query_context",
        remaining_space=80,
        min_span_length=2,
        max_span_length=4,
        min_num_spans=1,
        max_num_spans=2
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "Check query_context ordering for selective copying."
    out = obj(text)
    assert out["status"] == "ok"
    decoded_input = tokenizer.decode(out["input_ids"])
    # We expect [FUNC_0]... or similar instructions come before the [CONTEXT] marker
    context_pos = decoded_input.find(CONTEXT_TOKEN)
    func_pos = decoded_input.find("[FUNC_0")
    if func_pos != -1:
        assert func_pos < context_pos, "Instructions should appear before [CONTEXT] in query_context mode"
    # The label should have [RESULT_0]
    decoded_label = tokenizer.decode(out["label_ids"])
    assert "[RESULT_" in decoded_label

def test_selective_copying_context_query(tokenizer):
    """
    For 'context_query', the context should appear first, then instructions.
    """
    cfg = SelectiveCopyingConfig(
        formatting_type="context_query",
        remaining_space=80,
        min_span_length=2,
        max_span_length=4,
        min_num_spans=1,
        max_num_spans=2
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "Check context_query ordering for selective copying."
    out = obj(text)
    assert out["status"] == "ok"
    decoded_input = tokenizer.decode(out["input_ids"])
    context_pos = decoded_input.find(CONTEXT_TOKEN)
    func_pos = decoded_input.find("[FUNC_0")
    if func_pos != -1:
        assert context_pos < func_pos, "Context block should appear before instructions in context_query mode"
    decoded_label = tokenizer.decode(out["label_ids"])
    assert "[RESULT_" in decoded_label

def test_selective_copying_too_short(tokenizer):
    """If the input is shorter than min_span_length, no spans should be selected."""
    cfg = SelectiveCopyingConfig(
        formatting_type="query_context",
        remaining_space=50,
        min_span_length=10,
        max_span_length=12
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "Short text"
    out = obj(text)
    assert out["status"] == "ok"
    # label_ids should be empty if we didn't pick any spans
    assert len(out["label_ids"]) == 0, "No spans => empty label"

def test_selective_copying_unused_text(tokenizer):
    """Check leftover text logic for selective copying with small remaining space."""
    cfg = SelectiveCopyingConfig(
        formatting_type="context_query",
        remaining_space=3,
        min_span_length=2,
        max_span_length=4
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "One two three four"
    res = obj(text)
    assert res["status"] == "ok"
    leftover = res["unused_input_string"]
    # At least 'four' should be leftover if we used only 3 tokens
    assert "four" in leftover
    used_decoded = tokenizer.decode(res["input_ids"])
    assert "four" not in used_decoded


def test_selective_copying_chinese(tokenizer):
    """
    Test selective copying with Chinese text to see if spans are properly selected.
    """
    cfg = SelectiveCopyingConfig(
        formatting_type="query_context",
        remaining_space=40,
        min_span_length=2,
        max_span_length=3,
        min_num_spans=1,
        max_num_spans=1
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "你好 世界 你好 世界"
    res = obj(text)
    assert res["status"] == "ok"
    # Check label for at least one [RESULT_0]
    decoded_label = tokenizer.decode(res["label_ids"])
    assert "[RESULT_0]" in decoded_label

def test_selective_copying_hindi(tokenizer):
    """
    Test selective copying with Hindi text.
    """
    cfg = SelectiveCopyingConfig(
        formatting_type="context_query",
        remaining_space=50,
        min_span_length=3,
        max_span_length=4
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया नमस्ते फिर"
    res = obj(text)
    assert res["status"] == "ok"
    decoded_label = tokenizer.decode(res["label_ids"])
    # Expect a [RESULT_0] block
    assert "[RESULT_0]" in decoded_label

def test_selective_copying_long_dna(tokenizer):
    """
    Test selective copying with a repeated DNA sequence.
    """
    cfg = SelectiveCopyingConfig(
        formatting_type="query_context",
        remaining_space=60,
        min_span_length=4,
        max_span_length=8,
        min_num_spans=1,
        max_num_spans=2
    )
    obj = SelectiveCopyingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    dna_text = "AGTCAGTCAGTCAGTC"
    res = obj(dna_text)
    assert res["status"] == "ok"
    decoded_label = tokenizer.decode(res["label_ids"])
    # Should have a [RESULT_0] block with some substring of the DNA text
    assert "[RESULT_0]" in decoded_label
