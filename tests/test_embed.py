"""Tests for embedding backends.

Only the hashing backend is exercised here; the sentence-transformers backend
is covered by its import guard and otherwise needs a model download, which does
not belong in unit tests.
"""

import numpy as np
import pytest

from gdpr_rag.embed import HashingEmbedder, normalise


class TestNormalise:
    def test_rows_become_unit_length(self):
        vectors = normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_zero_vector_survives_instead_of_becoming_nan(self):
        vectors = normalise(np.array([[0.0, 0.0]]))
        assert not np.isnan(vectors).any()

    def test_single_vector_is_promoted_to_two_dimensions(self):
        assert normalise(np.array([3.0, 4.0])).shape == (1, 2)


class TestHashingEmbedder:
    def test_output_shape_matches_inputs_and_dimensions(self):
        embedder = HashingEmbedder(dimensions=64)
        assert embedder.encode(["one", "two", "three"]).shape == (3, 64)

    def test_vectors_are_unit_length(self):
        vectors = HashingEmbedder(dimensions=64).encode(["the data subject shall have the right"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_encoding_is_deterministic_across_instances(self):
        text = ["the right to erasure"]
        assert np.array_equal(
            HashingEmbedder(dimensions=64).encode(text),
            HashingEmbedder(dimensions=64).encode(text),
        )

    def test_shared_vocabulary_scores_higher_than_disjoint(self):
        embedder = HashingEmbedder(dimensions=1024)
        query, related, unrelated = embedder.encode(
            [
                "erasure of personal data",
                "the right to erasure of personal data without undue delay",
                "supervisory authority cooperation procedure",
            ]
        )
        assert float(query @ related) > float(query @ unrelated)

    def test_empty_text_gives_a_zero_vector_not_nan(self):
        vectors = HashingEmbedder(dimensions=32).encode([""])
        assert not np.isnan(vectors).any()
        assert np.allclose(vectors, 0.0)

    def test_name_records_the_configuration(self):
        assert HashingEmbedder(dimensions=256).name == "hashing-256"

    def test_invalid_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="dimensions must be positive"):
            HashingEmbedder(dimensions=0)
