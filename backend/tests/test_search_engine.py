"""
Unit tests for Web / Social Media Search Engine module and ConsentRegistrySearchProvider.
"""

import json
import os
import tempfile
import unittest
import cv2
import numpy as np
from app.search_engine import (
    CandidateVerificationService,
    ConsentRegistrySearchProvider,
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
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.imwrite(tmp_path, dummy_img)

            img_bytes, mime_type = optimize_image_for_upload(tmp_path, max_bytes=500 * 1024)
            self.assertIsInstance(img_bytes, bytes)
            self.assertLessEqual(len(img_bytes), 500 * 1024)
            self.assertIn(mime_type, ["image/jpeg", "image/png"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # --- ConsentRegistrySearchProvider Unit Tests ---

    def test_consent_registry_valid_candidates_returned(self):
        posts = [
            {
                "id": "post-001",
                "url": "https://www.linkedin.com/in/john-doe/posts/1",
                "image_url": "https://example.com/john.jpg",
                "platform": "LinkedIn",
                "owner": "John Doe"
            }
        ]
        provider = ConsentRegistrySearchProvider(posts_data=posts)
        response = provider.search_by_image("dummy_query.jpg")

        self.assertTrue(response.success)
        self.assertEqual(response.provider_name, "Consent Registry Search Provider")
        self.assertEqual(response.total_results, 1)

        candidate = response.candidates[0]
        self.assertEqual(candidate.url, "https://www.linkedin.com/in/john-doe/posts/1")
        self.assertEqual(candidate.source_domain, "linkedin.com")
        self.assertEqual(candidate.original_image_url, "https://example.com/john.jpg")
        self.assertIn("John Doe", candidate.title)

    def test_consent_registry_required_fields_validation(self):
        # Post missing 'image_url' and 'owner'
        posts = [
            {"id": "post-001", "url": "https://example.com/incomplete"},
            {
                "id": "post-002",
                "url": "https://x.com/user/1",
                "image_url": "https://example.com/img.jpg",
                "platform": "X",
                "owner": "User"
            }
        ]
        provider = ConsentRegistrySearchProvider(posts_data=posts)
        response = provider.search_by_image("dummy_query.jpg")

        self.assertTrue(response.success)
        # Invalid post #1 skipped, valid post #2 returned
        self.assertEqual(response.total_results, 1)
        self.assertEqual(response.candidates[0].url, "https://x.com/user/1")

    def test_consent_registry_unregistered_candidates_not_returned(self):
        posts = [
            {
                "id": "post-100",
                "url": "https://instagram.com/p/consented",
                "image_url": "https://example.com/photo.jpg",
                "platform": "Instagram",
                "owner": "Alice"
            }
        ]
        provider = ConsentRegistrySearchProvider(posts_data=posts)
        response = provider.search_by_image("dummy_query.jpg")

        urls = [c.url for c in response.candidates]
        self.assertIn("https://instagram.com/p/consented", urls)
        self.assertNotIn("https://unregistered-site.com/photo", urls)

    def test_consent_registry_multiple_candidates_processed(self):
        posts = [
            {
                "id": f"post-{i}",
                "url": f"https://platform{i}.com/post/{i}",
                "image_url": f"https://platform{i}.com/img/{i}.jpg",
                "platform": f"Platform{i}",
                "owner": f"User{i}"
            }
            for i in range(5)
        ]
        provider = ConsentRegistrySearchProvider(posts_data=posts)
        response = provider.search_by_image("dummy_query.jpg")

        self.assertTrue(response.success)
        self.assertEqual(response.total_results, 5)
        self.assertEqual(len(response.candidates), 5)

    def test_consent_registry_empty_registry_handled(self):
        provider = ConsentRegistrySearchProvider(posts_data=[])
        response = provider.search_by_image("dummy_query.jpg")

        self.assertTrue(response.success)
        self.assertEqual(response.total_results, 0)
        self.assertEqual(len(response.candidates), 0)

    def test_consent_registry_missing_file_handled(self):
        provider = ConsentRegistrySearchProvider(registry_file="non_existent_registry_file.json")
        response = provider.search_by_image("dummy_query.jpg")

        self.assertFalse(response.success)
        self.assertIn("not found", response.error_message)

    # --- Other Providers Unit Tests ---

    def test_mock_search_provider(self):
        provider = MockReverseImageSearchProvider()
        response = provider.search_by_image("dummy_image.jpg")

        self.assertTrue(response.success)
        self.assertEqual(response.provider_name, "Mock Search Provider")
        self.assertGreater(len(response.candidates), 0)

    def test_serpapi_missing_key_handling(self):
        provider = SerpApiGoogleLensProvider(api_key="")
        response = provider.search_by_image("dummy_image.jpg")

        self.assertFalse(response.success)
        self.assertIn("SERPAPI_API_KEY is not set", response.error_message)

    def test_candidate_verification_service(self):
        candidates = [
            SearchResultCandidate(title="Insta", url="https://instagram.com/p/1", source_domain="instagram.com", is_social_media=True),
            SearchResultCandidate(title="News", url="https://news.com/a/1", source_domain="news.com", is_social_media=False),
            SearchResultCandidate(title="LinkedIn", url="https://linkedin.com/in/1", source_domain="linkedin.com", is_social_media=True),
        ]

        social_only = CandidateVerificationService.filter_social_media_candidates(candidates)
        self.assertEqual(len(social_only), 2)

        ranked = CandidateVerificationService.rank_candidates(candidates)
        self.assertEqual(len(ranked), 3)


if __name__ == "__main__":
    unittest.main()
