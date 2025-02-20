import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.infilling import InfillingConfig, InfillingObjective

@pytest.fixture(scope="module")
def tokenizer():
    return Tokenizer()

def test_infilling_no_mask(tokenizer):
    """With corruption_rate=0, we expect pass-through behavior."""
    config = InfillingConfig(
        corruption_rate=0.0,
        tokens_per_mask=1,
        max_mask_spans=10,
        infilling_prefix="[mask_",
        separator=" "
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "No masking should leave text alone."
    result = obj(text)
    assert result["status"] == "ok"
    # input_ids == label_ids
    assert np.array_equal(result["input_ids"], result["label_ids"])
    decoded_input = tokenizer.decode(result["input_ids"])
    assert decoded_input == text

def test_infilling_some_mask(tokenizer):
    """Test scenario with partial masking."""
    config = InfillingConfig(
        corruption_rate=0.5,
        tokens_per_mask=2,
        max_mask_spans=5,
        infilling_prefix="[mask_",
        separator="~"
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "Partial masking test for the infilling objective."
    result = obj(text)
    assert result["status"] == "ok"
    decoded_input = tokenizer.decode(result["input_ids"])
    decoded_label = tokenizer.decode(result["label_ids"])
    # Expect placeholders in both input & label
    assert "[mask_" in decoded_input
    assert "[mask_" in decoded_label
    assert decoded_input.count("[mask_") == decoded_label.count("[mask_"), \
        "Each placeholder in the input should appear in label."

def test_infilling_full_mask(tokenizer):
    """Check 100% corruption. All tokens should be replaced in input, appended in label."""
    config = InfillingConfig(
        corruption_rate=1.0,
        tokens_per_mask=1,
        max_mask_spans=50,
        infilling_prefix="[mask_",
        separator=" "
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "Completely masked text."
    result = obj(text)
    assert result["status"] == "ok"
    inp = tokenizer.decode(result["input_ids"])
    lbl = tokenizer.decode(result["label_ids"])
    # Input should have placeholders only
    assert "[mask_" in inp
    # Label is a sequence of placeholders plus tokens
    assert "[mask_" in lbl
    # Number of placeholders should match total tokens
    total_toks = len(tokenizer.encode(text))
    input_placeholders = inp.count("[mask_")
    label_placeholders = lbl.count("[mask_")
    assert input_placeholders == label_placeholders == total_toks

def test_infilling_remaining_space(tokenizer):
    """Check leftover logic with limited space."""
    cfg = InfillingConfig(
        remaining_space=4,
        corruption_rate=0.5,
        tokens_per_mask=1
    )
    obj = InfillingObjective(cfg)
    text = "One two three four five"
    res = obj(text)
    assert res["status"] == "ok"
    # Only 'One two three four' should be used. 'five' leftover
    leftover = res["unused_input_string"]
    assert "five" in leftover
    used_decoded = tokenizer.decode(res["input_ids"])
    assert "five" not in used_decoded


def test_infilling_chinese_text(tokenizer):
    """
    Test infilling with Chinese text.
    """
    config = InfillingConfig(
        corruption_rate=0.4,
        tokens_per_mask=1,
        infilling_prefix="[mask_",
        separator=" "
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "你好，世界。让我们测试infilling。"
    res = obj(text)
    assert res["status"] == "ok"
    decoded_input = tokenizer.decode(res["input_ids"])
    decoded_label = tokenizer.decode(res["label_ids"])
    # We expect some placeholders in both input & label if corruption_rate > 0
    if config.corruption_rate > 0:
        assert "[mask_" in decoded_input
        assert "[mask_" in decoded_label

def test_infilling_hindi_text(tokenizer):
    """
    Test infilling with Hindi text.
    """
    config = InfillingConfig(
        corruption_rate=0.3,
        tokens_per_mask=2
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया! यह एक और परीक्षण है।"
    res = obj(text)
    assert res["status"] == "ok"
    decoded_label = tokenizer.decode(res["label_ids"])
    assert text not in decoded_label, (
        "Label string contains placeholders plus masked tokens, so it won't match exactly. "
        "But we must confirm no error occurs."
    )

def test_infilling_dna_sequence(tokenizer):
    """
    Test infilling with a DNA sequence repeated to ensure placeholders are inserted as expected.
    """
    dna_text = "AGTCAGTCAGTCAGTC"
    config = InfillingConfig(
        corruption_rate=0.5,
        tokens_per_mask=2,
        max_mask_spans=4
    )
    obj = InfillingObjective(config)
    obj.set_tokenizer(tokenizer)
    res = obj(dna_text)
    assert res["status"] == "ok"
    decoded_input = tokenizer.decode(res["input_ids"])
    # Check presence of placeholders
    if config.corruption_rate > 0:
        assert "[mask_" in decoded_input
