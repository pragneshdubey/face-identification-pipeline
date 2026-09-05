"""
Unit tests for FastAPI Backend API endpoints & Security Hardening.
"""

import os
import unittest
from fastapi.testclient import TestClient

from app.api import app, PROJECT_ROOT


class TestAPIEndpoints(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["embedding_dimensions"], 512)
        self.assertEqual(data["default_threshold"], 0.5)

    def test_verify_endpoint_mock_provider(self):
        sample_img = os.path.join(PROJECT_ROOT, "data", "test_images", "einstein_demo.jpg")
        if not os.path.exists(sample_img):
            self.skipTest(f"Sample image '{sample_img}' not found.")

        response = self.client.post(
            "/api/verify",
            data={
                "provider": "mock",
                "threshold": 0.5,
                "image_name": "einstein_demo.jpg"
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["face_detected"])
        self.assertEqual(data["threshold"], 0.5)
        self.assertIn("logs", data)

    def test_arbitrary_path_traversal_rejection(self):
        # Attempt path traversal or arbitrary system file access via image_name
        response = self.client.post(
            "/api/verify",
            data={
                "provider": "mock",
                "image_name": "../../requirements.txt"
            }
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or unauthorized test image name", response.json()["detail"])

    def test_unsupported_file_extension_rejection(self):
        # Upload a file with unsupported extension (.txt)
        response = self.client.post(
            "/api/verify",
            files={"file": ("malicious.txt", b"hello world", "text/plain")},
            data={"provider": "mock"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json()["detail"])

    def test_empty_file_upload_rejection(self):
        # Upload an empty file
        response = self.client.post(
            "/api/verify",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            data={"provider": "mock"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Uploaded file is empty", response.json()["detail"])

    def test_oversized_file_upload_rejection(self):
        # Upload a file larger than 10MB (10MB + 100 bytes)
        large_bytes = b"0" * (10 * 1024 * 1024 + 100)
        response = self.client.post(
            "/api/verify",
            files={"file": ("huge.jpg", large_bytes, "image/jpeg")},
            data={"provider": "mock"}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("exceeds maximum limit of 10 MB", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
