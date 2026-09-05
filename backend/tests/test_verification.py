"""
Unit tests for Candidate Face Verification engine using synthetic embeddings and mocks.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from app.face_engine import normalize_embedding
from app.search_engine import (
    CandidateFaceVerifier,
    CandidateVerificationResult,
    SearchResultCandidate,
)


class TestCandidateFaceVerification(unittest.TestCase):

    def setUp(self):
        # Create synthetic 128D query embedding
        self.query_emb = normalize_embedding(np.random.randn(128).astype(np.float32))

        # Mock FaceIdentificationEngine
        self.mock_face_engine = MagicMock()
        self.verifier = CandidateFaceVerifier(
            face_engine=self.mock_face_engine,
            face_threshold=0.6,
            stop_on_first_match=True
        )

    def test_missing_image_url_rejection(self):
        candidate = SearchResultCandidate(
            title="No Image Post",
            url="https://example.com/post1",
            source_domain="example.com",
            thumbnail_url=None,
            original_image_url=None
        )

        res = self.verifier.verify_candidate(self.query_emb, candidate)

        self.assertFalse(res.is_verified_match)
        self.assertEqual(res.verification_status, "REJECTED_NO_IMAGE_URL")
        self.assertIn("No candidate image URL available", res.rejection_reason)

    @patch.object(CandidateFaceVerifier, "download_candidate_image")
    def test_download_failure_rejection(self, mock_download):
        mock_download.return_value = (None, "HTTP 404 Not Found")

        candidate = SearchResultCandidate(
            title="Broken Image Link",
            url="https://example.com/post2",
            source_domain="example.com",
            thumbnail_url="https://example.com/broken.jpg"
        )

        res = self.verifier.verify_candidate(self.query_emb, candidate)

        self.assertFalse(res.is_verified_match)
        self.assertEqual(res.verification_status, "REJECTED_DOWNLOAD_ERROR")
        self.assertIn("download failed", res.rejection_reason)

    @patch.object(CandidateFaceVerifier, "download_candidate_image")
    def test_no_face_detected_rejection(self, mock_download):
        # Return dummy image
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_download.return_value = (dummy_img, None)

        # Mock face engine returning 0 faces
        self.mock_face_engine.extract_faces.return_value = []

        candidate = SearchResultCandidate(
            title="Landscape Image",
            url="https://example.com/post3",
            source_domain="example.com",
            thumbnail_url="https://example.com/landscape.jpg"
        )

        res = self.verifier.verify_candidate(self.query_emb, candidate)

        self.assertFalse(res.is_verified_match)
        self.assertEqual(res.verification_status, "REJECTED_NO_FACE")
        self.assertIn("No faces detected", res.rejection_reason)

    @patch.object(CandidateFaceVerifier, "download_candidate_image")
    def test_low_similarity_rejection(self, mock_download):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_download.return_value = (dummy_img, None)

        # Different face embedding (orthogonal)
        different_emb = normalize_embedding(np.random.randn(128).astype(np.float32))

        self.mock_face_engine.extract_faces.return_value = [
            {"bbox": (10, 10, 50, 50), "embedding": different_emb}
        ]

        candidate = SearchResultCandidate(
            title="Different Person Post",
            url="https://example.com/post4",
            source_domain="example.com",
            thumbnail_url="https://example.com/person2.jpg"
        )

        res = self.verifier.verify_candidate(self.query_emb, candidate)

        self.assertFalse(res.is_verified_match)
        self.assertEqual(res.verification_status, "REJECTED_LOW_SIMILARITY")

    @patch.object(CandidateFaceVerifier, "download_candidate_image")
    def test_verified_match_success(self, mock_download):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_download.return_value = (dummy_img, None)

        # Matching face embedding (identical to query_emb)
        matching_emb = self.query_emb.copy()

        self.mock_face_engine.extract_faces.return_value = [
            {"bbox": (15, 20, 80, 85), "embedding": matching_emb}
        ]

        candidate = SearchResultCandidate(
            title="Target Person Post on LinkedIn",
            url="https://linkedin.com/in/target-person",
            source_domain="linkedin.com",
            thumbnail_url="https://linkedin.com/photo.jpg",
            is_social_media=True
        )

        res = self.verifier.verify_candidate(self.query_emb, candidate)

        self.assertTrue(res.is_verified_match)
        self.assertEqual(res.verification_status, "VERIFIED_MATCH")
        self.assertAlmostEqual(res.face_similarity_score, 1.0, places=4)
        self.assertEqual(res.best_face_bbox, (15, 20, 80, 85))

    @patch.object(CandidateFaceVerifier, "download_candidate_image")
    def test_verify_search_candidates_early_stop(self, mock_download):
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_download.return_value = (dummy_img, None)

        cand1 = SearchResultCandidate(title="Cand 1 (Match)", url="https://example.com/1", source_domain="example.com", thumbnail_url="https://example.com/1.jpg")
        cand2 = SearchResultCandidate(title="Cand 2 (Unverified)", url="https://example.com/2", source_domain="example.com", thumbnail_url="https://example.com/2.jpg")

        # Mock extract_faces to return matching embedding for candidate 1
        self.mock_face_engine.extract_faces.return_value = [
            {"bbox": (0, 0, 10, 10), "embedding": self.query_emb.copy()}
        ]

        results = self.verifier.verify_search_candidates(self.query_emb, [cand1, cand2])

        # Should stop after first verified match
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_verified_match)


if __name__ == "__main__":
    unittest.main()

