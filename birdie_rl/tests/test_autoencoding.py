import pytest
import numpy as np
from modeling.tokenizer import Tokenizer
from objectives.autoencoding import AutoencodingConfig, AutoencodingObjective

# We assume there's a fixture named 'tokenizer' in conftest or test_utils,
# but if not, we can create our own here.

def test_autoencoding_no_mask(tokenizer):
    """
    Verify that with corruption_rate=0 we get a pass-through 
    of the original text (input_ids == label_ids).
    """
    config = AutoencodingConfig(
        corruption_rate=0.0,
        tokens_per_mask=1,
        mask_prefix="[mask_",
        shuffle=False,
        separator=" "
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    
    text = "This is a test."
    output = objective(text)

    print("\n[DEBUG] Autoencoding no_mask =>")
    print("  Original text:", text)
    print("  Output:", output)

    assert output["status"] == "ok"
    assert np.array_equal(output["input_ids"], output["label_ids"]), "Expected no corruption"
    decoded_input = tokenizer.decode(output["input_ids"])
    decoded_label = tokenizer.decode(output["label_ids"])
    assert decoded_input == text
    assert decoded_label == text

def test_autoencoding_partial_mask_and_shuffle(tokenizer):
    """
    Test partial masking with shuffle, ensuring placeholders exist.
    """
    config = AutoencodingConfig(
        corruption_rate=0.5,
        tokens_per_mask=2,
        max_mask_spans=5,
        mask_prefix="[mask_",
        shuffle=True,
        separator="~"
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    text = "Partial shuffle test. Some tokens should be masked."
    result = objective(text)
    assert result["status"] == "ok"
    decoded_inp = tokenizer.decode(result["input_ids"])
    decoded_lbl = tokenizer.decode(result["label_ids"])
    assert "[mask_" in decoded_inp
    assert decoded_lbl == text

def test_autoencoding_no_mask(tokenizer):
    """
    Duplicate named test, but let's keep it. Check pass-through again.
    """
    config = AutoencodingConfig(
        corruption_rate=0.0,
        tokens_per_mask=1,
        mask_prefix="[mask_",
        shuffle=False,
        separator=" "
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    
    text = "This is a test."
    output = objective(text)
    assert output["status"] == "ok"
    assert np.array_equal(output["input_ids"], output["label_ids"])
    decoded_input = tokenizer.decode(output["input_ids"])
    assert decoded_input == text

def test_autoencoding_full_mask(tokenizer):
    """
    With corruption_rate=1, everything is replaced in the input.
    """
    config = AutoencodingConfig(
        corruption_rate=1.0,
        tokens_per_mask=1,
        mask_prefix="[mask_",
        shuffle=False,
        separator=" "
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)

    text = "Testing full mask scenario."
    output = objective(text)
    assert output["status"] == "ok"
    decoded_input = tokenizer.decode(output["input_ids"])
    decoded_label = tokenizer.decode(output["label_ids"])
    assert decoded_label == text
    assert "[mask_" in decoded_input
    original_tokens = tokenizer.encode(text)
    placeholder_count = decoded_input.count("[mask_")
    assert placeholder_count == len(original_tokens)

def test_autoencoding_partial_mask_and_shuffle(tokenizer):
    """
    Another partial shuffle test - placeholders must appear.
    """
    config = AutoencodingConfig(
        corruption_rate=0.5,
        tokens_per_mask=2,
        max_mask_spans=5,
        mask_prefix="[mask_",
        shuffle=True,
        separator="~"
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    text = "Partial shuffle test. Some tokens should be masked."
    result = objective(text)
    assert result["status"] == "ok"
    decoded_inp = tokenizer.decode(result["input_ids"])
    decoded_lbl = tokenizer.decode(result["label_ids"])
    assert "[mask_" in decoded_inp
    assert decoded_lbl == text

def test_autoencoding_space_checks(tokenizer):
    """
    Check leftover text usage with small remaining space.
    """
    config = AutoencodingConfig(
        remaining_space=5,
        corruption_rate=0.3,
        tokens_per_mask=1
    )
    objective = AutoencodingObjective(config)
    text = "One two three four five six seven eight"
    result = objective(text)
    assert result["status"] == "ok"
    leftover = result["unused_input_string"]
    assert leftover, "Expected leftover text"

def test_autoencoding_chinese_text(tokenizer):
    """
    Chinese text check for placeholders and correctness.
    """
    config = AutoencodingConfig(
        corruption_rate=0.5,
        tokens_per_mask=1,
        shuffle=False
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    text = "你好世界！这是一个测试。"
    output = objective(text)
    assert output["status"] == "ok"
    decoded_input = tokenizer.decode(output["input_ids"])
    decoded_label = tokenizer.decode(output["label_ids"])
    if config.corruption_rate > 0:
        assert "[mask_" in decoded_input
    assert decoded_label == text

def test_autoencoding_hindi_text(tokenizer):
    """
    Hindi text check.
    """
    config = AutoencodingConfig(
        corruption_rate=0.3,
        tokens_per_mask=2,
        max_mask_spans=2,
        shuffle=True
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    text = "नमस्ते दुनिया यह एक परीक्षण है"
    output = objective(text)
    assert output["status"] == "ok"
    decoded_label = tokenizer.decode(output["label_ids"])
    assert decoded_label == text

def test_autoencoding_long_dna_sequence(tokenizer):
    """
    Check large repeated text.
    """
    dna_text = "AGTC" * 50
    config = AutoencodingConfig(
        corruption_rate=0.1,
        tokens_per_mask=5,
        max_mask_spans=10,
        shuffle=False
    )
    objective = AutoencodingObjective(config)
    objective.set_tokenizer(tokenizer)
    output = objective(dna_text)
    assert output["status"] == "ok"
    decoded_label = tokenizer.decode(output["label_ids"])
    assert decoded_label == dna_text
