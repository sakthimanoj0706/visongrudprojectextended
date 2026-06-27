import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.similarity import cosine_similarity, is_match

def test_identical_similarity():
    vec = np.random.rand(128)
    sim = cosine_similarity(vec, vec)
    assert np.isclose(sim, 1.0), f"Expected 1.0, got {sim}"

def test_opposite_similarity():
    vec = np.random.rand(128)
    sim = cosine_similarity(vec, -vec)
    assert np.isclose(sim, -1.0), f"Expected -1.0, got {sim}"

def test_orthogonal_similarity():
    # Make two orthogonal vectors
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    sim = cosine_similarity(vec_a, vec_b)
    assert np.isclose(sim, 0.0), f"Expected 0.0, got {sim}"

def test_is_match():
    assert is_match(0.75, threshold=0.60) is True
    assert is_match(0.55, threshold=0.60) is False
    assert is_match(0.60, threshold=0.60) is True
