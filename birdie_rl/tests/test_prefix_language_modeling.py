import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.prefix_language_modeling import (
    PrefixLanguageModelingConfig,
    PrefixLanguageModelingObjective
)

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_prefix_language_modeling_short_input(tokenizer):
    """If we have fewer than 2 tokens, the prefix and suffix should be identical."""
    config = PrefixLanguageModelingConfig(
        prefix_fraction=0.7,
        remaining_space=10
    )
    obj = PrefixLanguageModelingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "Hi"
    result = obj(text)
    assert result["status"] == "ok"
    # input_ids == label_ids for very short text
    assert np.array_equal(result["input_ids"], result["label_ids"])

def test_prefix_language_modeling_split(tokenizer):
    """For longer text, we expect a prefix portion in input_ids and a suffix portion in label_ids."""
    config = PrefixLanguageModelingConfig(
        prefix_fraction=0.5,
        remaining_space=100
    )
    obj = PrefixLanguageModelingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "Token1 Token2 Token3 Token4"
    result = obj(text)
    assert result["status"] == "ok"
    prefix_ids = result["input_ids"]
    label_ids = result["label_ids"]
    # We expect about half the tokens in prefix, half in label
    total_tokens = len(obj.tokenizer.encode(text))
    assert len(prefix_ids) + len(label_ids) == total_tokens
    assert len(prefix_ids) > 0
    assert len(label_ids) > 0

def test_prefix_language_modeling_leftover(tokenizer):
    """Check leftover text is properly set if we run out of space for the prefix + suffix."""
    config = PrefixLanguageModelingConfig(
        prefix_fraction=0.7,
        remaining_space=5
    )
    obj = PrefixLanguageModelingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "One two three four five six"
    result = obj(text)
    assert result["status"] == "ok"
    leftover_str = result["unused_input_string"]
    used_decoded = tokenizer.decode(result["input_ids"])
    used_decoded += tokenizer.decode(result["label_ids"])
    # We only used 5 tokens total
    leftover_toks = leftover_str.split()
    for tok in leftover_toks:
        assert tok not in used_decoded, "Leftover tokens must not appear in the used portion."

def test_prefix_language_modeling_chinese(tokenizer):
    """
    Test prefix language modeling with Chinese text.
    """
    cfg = PrefixLanguageModelingConfig(
        prefix_fraction=0.5,
        remaining_space=12
    )
    obj = PrefixLanguageModelingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "你好 世界"
    res = obj(text)
    assert res["status"] == "ok"
    # Prefix is half, suffix is half (approx)
    decoded_prefix = tokenizer.decode(res["input_ids"])
    decoded_suffix = tokenizer.decode(res["label_ids"])
    # Combined
    combined = decoded_prefix + decoded_suffix
    assert "你好" in combined or "世界" in combined

def test_prefix_language_modeling_hindi(tokenizer):
    """
    Test prefix language modeling with Hindi text.
    """
    cfg = PrefixLanguageModelingConfig(
        prefix_fraction=0.6,
        remaining_space=8
    )
    obj = PrefixLanguageModelingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया नमस्ते"
    res = obj(text)
    assert res["status"] == "ok"
    # Should have partial prefix, partial suffix
    decoded_prefix = tokenizer.decode(res["input_ids"])
    decoded_suffix = tokenizer.decode(res["label_ids"])
    assert len(decoded_prefix) > 0
    assert len(decoded_suffix) > 0

def test_prefix_language_modeling_dna(tokenizer):
    """
    Test prefix language modeling with DNA sequences.
    """
    dna_text = "AGTCAGTCAGTC"
    cfg = PrefixLanguageModelingConfig(
        prefix_fraction=0.3,
        remaining_space=10
    )
    obj = PrefixLanguageModelingObjective(cfg)
    obj.set_tokenizer(tokenizer)
    res = obj(dna_text)
    assert res["status"] == "ok"
    prefix_decoded = tokenizer.decode(res["input_ids"])
    suffix_decoded = tokenizer.decode(res["label_ids"])
    # Both prefix and suffix should be non-empty if the text is long enough
    assert prefix_decoded != ""
    assert suffix_decoded != ""
