"""
Unit tests for Local Blockchain Module.
"""

import unittest
from app.blockchain import (
    Block,
    LocalBlockchainService,
    compute_sha256_fingerprint,
)


class TestLocalBlockchain(unittest.TestCase):

    def setUp(self):
        self.blockchain = LocalBlockchainService()
        self.sample_payload = {
            "title": "Sample Verified Post",
            "source_url": "https://www.linkedin.com/in/sample",
            "source_domain": "linkedin.com",
            "similarity_score": 0.85,
            "verification_status": "VERIFIED_MATCH",
            "timestamp": "2026-09-04T00:00:00Z"
        }

    def test_genesis_block(self):
        self.assertEqual(len(self.blockchain.chain), 1)
        genesis = self.blockchain.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.previous_hash, "0" * 64)
        is_valid, err = self.blockchain.verify_chain()
        self.assertTrue(is_valid)
        self.assertIsNone(err)

    def test_add_and_retrieve_record(self):
        tx_id = self.blockchain.add_record(self.sample_payload)
        self.assertIsInstance(tx_id, str)
        self.assertEqual(len(tx_id), 64)

        retrieved = self.blockchain.get_record(tx_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["title"], self.sample_payload["title"])
        self.assertEqual(retrieved["source_domain"], "linkedin.com")
        self.assertIn("fingerprint", retrieved)

    def test_successful_fingerprint_verification(self):
        tx_id = self.blockchain.add_record(self.sample_payload)

        # Re-verify matching candidate data
        status, score = self.blockchain.verify_record(tx_id, self.sample_payload)
        self.assertEqual(status, "VERIFIED")
        self.assertEqual(score, 1.0)

    def test_tampered_content_verification_failure(self):
        tx_id = self.blockchain.add_record(self.sample_payload)

        # Modified payload data
        tampered_payload = dict(self.sample_payload)
        tampered_payload["similarity_score"] = 0.99
        tampered_payload["title"] = "Tampered Title"

        status, score = self.blockchain.verify_record(tx_id, tampered_payload)
        self.assertEqual(status, "TAMPERED")
        self.assertEqual(score, 0.0)

    def test_chain_integrity_tamper_detection(self):
        tx_id1 = self.blockchain.add_record(self.sample_payload)
        tx_id2 = self.blockchain.add_record({"title": "Second Post", "url": "https://example.com"})

        # Initial chain integrity check passes
        is_valid, _ = self.blockchain.verify_chain()
        self.assertTrue(is_valid)

        # Mutate block data in an earlier block directly
        block1 = self.blockchain.get_block(tx_id1)
        block1.data["title"] = "Hacked Title"

        # Chain integrity check must fail
        is_valid, err = self.blockchain.verify_chain()
        self.assertFalse(is_valid)
        self.assertIn("hash mismatch", err)

    def test_chain_integrity_broken_link_detection(self):
        tx_id1 = self.blockchain.add_record(self.sample_payload)
        tx_id2 = self.blockchain.add_record({"title": "Second Post", "url": "https://example.com"})

        block2 = self.blockchain.get_block(tx_id2)
        # Corrupt previous_hash link
        block2.previous_hash = "f" * 64

        is_valid, err = self.blockchain.verify_chain()
        self.assertFalse(is_valid)
        self.assertIn("previous_hash does not match", err)


if __name__ == "__main__":
    unittest.main()

