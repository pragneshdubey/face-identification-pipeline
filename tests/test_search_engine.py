"""
Unit tests for Web / Social Media Search Engine module.
"""

import os
import tempfile
import unittest
import cv2
import numpy as np
from app.search_engine import (
    CandidateVerificationService,
    MockReverseImageSearchProvider,
    SearchResultCandidate,
    SerpApiGoogleLensProvider,
    extract_domain,
    is_social_media_domain,
    optimize_image_for_upload,
)


class TestSearchEngine(unittest.TestCase):

    def test_extract_domain(self):
        self.assertEqual(extract_domain("https://www.instagram.com/p/123"), "instagram.com")
        self.assertEqual(extract_domain("http://subdomain.linkedin.com/in/test"), "subdomain.linkedin.com")
        self.assertEqual(extract_domain("https://x.com/user/status/456"), "x.com")
        self.assertEqual(extract_domain(""), "")

    def test_is_social_media_domain(self):
        self.assertTrue(is_social_media_domain("instagram.com"))
        self.assertTrue(is_social_media_domain("sub.instagram.com"))
        self.assertTrue(is_social_media_domain("x.com"))
        self.assertTrue(is_social_media_domain("linkedin.com"))
        self.assertFalse(is_social_media_domain("example-news.com"))
        self.assertFalse(is_social_media_domain(""))

    def test_optimize_image_for_upload(self):
        # Create a temporary image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create a 500x500 dummy image
            dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.imwrite(tmp_path, dummy_img)

            img_bytes, mime_type = optimize_image_for_upload(tmp_path, max_bytes=500 * 1024)
            self.assertIsInstance(img_bytes, bytes)
            self.assertLessEqual(len(img_bytes), 500 * 1024)
            self.assertIn(mime_type, ["image/jpeg", "image/png"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_mock_search_provider(self):
        provider = MockReverseImageSearchProvider()
        response = provider.search_by_image("dummy_image.jpg")

        self.assertTrue(response.success)
        self.assertEqual(response.provider_name, "Mock Search Provider")
        self.assertGreater(len(response.candidates), 0)

        social_candidates = [c for c in response.candidates if c.is_social_media]
        self.assertGreater(len(social_candidates), 0)

    def test_serpapi_missing_key_handling(self):
        provider = SerpApiGoogleLensProvider(api_key="")
        response = provider.search_by_image("dummy_image.jpg")

        self.assertFalse(response.success)
        self.assertIn("SERPAPI_API_KEY is not set", response.error_message)

    def test_serpapi_missing_local_file_handling(self):
        provider = SerpApiGoogleLensProvider(api_key="fake_key_123")
        response = provider.search_by_image("non_existent_image_12345.jpg")

        self.assertFalse(response.success)
        self.assertIn("not found", response.error_message)

    def test_candidate_verification_service(self):
        candidates = [
            SearchResultCandidate(title="Insta", url="https://instagram.com/p/1", source_domain="instagram.com", is_social_media=True),
            SearchResultCandidate(title="News", url="https://news.com/a/1", source_domain="news.com", is_social_media=False),
            SearchResultCandidate(title="LinkedIn", url="https://linkedin.com/in/1", source_domain="linkedin.com", is_social_media=True),
        ]

        social_only = CandidateVerificationService.filter_social_media_candidates(candidates)
        self.assertEqual(len(social_only), 2)
        self.assertTrue(all(c.is_social_media for c in social_only))

        ranked = CandidateVerificationService.rank_candidates(candidates)
        self.assertEqual(len(ranked), 3)
        self.assertGreaterEqual(ranked[0].relevance_score, ranked[-1].relevance_score)


if __name__ == "__main__":
    unittest.main()

