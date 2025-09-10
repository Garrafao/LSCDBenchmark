import sys
sys.path.insert(0, "..")
import pytest
from src.utils.truncation import tokenize_truncate_decode
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("sachinn1/xl-durel")


def test_basic_case(tokenizer):
    sentence = "This is a simple test."
    positions = (10, 16)  # "simple"
    result = tokenize_truncate_decode(sentence, positions, tokenizer)
    assert "<t>" in result and "</t>" in result
    assert "simple" in result


def test_truncation(tokenizer):
    sentence = " ".join(["word"] * 500)
    positions = (10, 15)
    max_seq_len = 128
    result = tokenize_truncate_decode(sentence, positions, tokenizer, max_seq_len)
    assert len(result.split()) <= max_seq_len
    assert "word" in result



def test_long_sentence_truncation_preserves_target(tokenizer):
    # Create a sentence with 200 tokens
    sentence = " ".join([f"word{i}" for i in range(200)])
    
    # Choose a target word somewhere in the middle (safe position)
    target_word = "word100"
    start = sentence.index(target_word)
    end = start + len(target_word)

    positions = (start, end)  # target is "word100"

    # Run function
    result = tokenize_truncate_decode(sentence, positions, tokenizer, max_seq_len=128)

    # Assertions
    tokens = tokenizer.tokenize(result)

    assert len(tokens) <= 128, "Output should be truncated to <= 128 tokens"
    assert target_word in result, "Target word must be preserved after truncation"
    assert "<t>" in result and "</t>" in result, "Special tokens must wrap target word"
