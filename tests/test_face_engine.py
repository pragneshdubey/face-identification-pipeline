"""
Unit tests for Face Identification Engine core vector math and matching logic.
Uses synthetic vectors so tests do not require downloading ONNX models.
"""

import unittest
import numpy as np
from app.face_engine import (
    FaceIdentificationEngine,
    RecognitionResult,
    compute_cosine_similarity,
    find_best_match,
    normalize_embedding,
)


class TestVectorMathAndMatching(unittest.TestCase):

    def test_normalize_embedding(self):
        # 1. Normal vector
        vec = np.array([3.0, 4.0], dtype=np.float32)
        norm_vec = normalize_embedding(vec)
        self.assertAlmostEqual(np.linalg.norm(norm_vec), 1.0, places=5)
        np.testing.assert_allclose(norm_vec, [0.6, 0.8], rtol=1e-5)

        # 2. Zero vector
        zero_vec = np.array([0.0, 0.0], dtype=np.float32)
        norm_zero = normalize_embedding(zero_vec)
        np.testing.assert_array_equal(norm_zero, zero_vec)

    def test_compute_cosine_similarity(self):
        # Identical unit vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v2), 1.0, places=5)

        # Orthogonal vectors
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v3), 0.0, places=5)

        # Opposite vectors
        v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(compute_cosine_similarity(v1, v4), -1.0, places=5)

        # Zero vector handling
        zero_v = np.zeros(3, dtype=np.float32)
        self.assertEqual(compute_cosine_similarity(v1, zero_v), 0.0)

    def test_find_best_match_empty_db(self):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        known_embeddings = {}

        result = find_best_match(query_emb, known_embeddings, threshold=0.5)

        self.assertFalse(result.is_match)
        self.assertIsNone(result.person_id)
        self.assertEqual(result.similarity_score, 0.0)
        self.assertIn("No known identities", result.error_message)

    def test_find_best_match_above_threshold(self):
        query_emb = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        query_emb = normalize_embedding(query_emb)

        alice_emb = normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        bob_emb = normalize_embedding(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        known = {"alice": alice_emb, "bob": bob_emb}

        result = find_best_match(query_emb, known, threshold=0.5)

        self.assertTrue(result.is_match)
        self.assertEqual(result.person_id, "alice")
        self.assertGreater(result.similarity_score, 0.5)
        self.assertIsNone(result.error_message)

    def test_find_best_match_below_threshold(self):
        query_emb = normalize_embedding(np.array([0.5, 0.5, 0.0], dtype=np.float32))
        alice_emb = normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        known = {"alice": alice_emb}

        # High threshold requiring 0.9 similarity
        result = find_best_match(query_emb, known, threshold=0.9)

        self.assertFalse(result.is_match)
        self.assertIsNone(result.person_id)
        self.assertIn("No match found above threshold", result.error_message)

    def test_configurable_threshold(self):
        query = normalize_embedding(np.array([1.0, 0.5, 0.0], dtype=np.float32))
        target = normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        known = {"target_person": target}

        sim_score = compute_cosine_similarity(query, target)

        # Loose threshold passes
        res_loose = find_best_match(query, known, threshold=sim_score - 0.05)
        self.assertTrue(res_loose.is_match)
        self.assertEqual(res_loose.person_id, "target_person")

        # Strict threshold fails
        res_strict = find_best_match(query, known, threshold=sim_score + 0.05)
        self.assertFalse(res_strict.is_match)
        self.assertIsNone(res_strict.person_id)

    def test_load_image_non_existent(self):
        img, err = FaceIdentificationEngine.load_image("non_existent_file_12345.jpg")
        self.assertIsNone(img)
        self.assertIn("not found", err)


if __name__ == "__main__":
    unittest.main()

