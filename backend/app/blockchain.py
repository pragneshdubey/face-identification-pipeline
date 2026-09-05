"""
Blockchain Module for Verifiable, Tamper-Evident Record Storage.

Provides a local SHA-256 linked blockchain and an abstract adapter interface
allowing future pluggability for public blockchain testnets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple


def compute_sha256_fingerprint(data: Any) -> str:
    """
    Computes a deterministic SHA-256 fingerprint hex string for arbitrary data.
    Excludes self-referential 'fingerprint' key if dict is passed.
    
    Args:
        data: Dict, string, or bytes input.
        
    Returns:
        64-character SHA-256 hex digest string.
    """
    if isinstance(data, dict):
        clean_data = {k: v for k, v in data.items() if k != "fingerprint"}
        json_str = json.dumps(clean_data, sort_keys=True, separators=(',', ':'))
        raw_bytes = json_str.encode("utf-8")
    elif isinstance(data, bytes):
        raw_bytes = data
    elif isinstance(data, str):
        raw_bytes = data.encode("utf-8")
    else:
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        raw_bytes = json_str.encode("utf-8")

    return hashlib.sha256(raw_bytes).hexdigest()


@dataclass
class Block:
    """Dataclass representing a block in the local blockchain."""
    index: int
    timestamp: float
    data: Dict[str, Any]
    previous_hash: str
    hash: str = ""

    def calculate_hash(self) -> str:
        """
        Calculates the SHA-256 hash of the block contents.
        
        Returns:
            64-character SHA-256 hex string.
        """
        canonical_data = json.dumps(self.data, sort_keys=True, separators=(',', ':'))
        block_content = f"{self.index}{self.timestamp:.6f}{self.previous_hash}{canonical_data}"
        return hashlib.sha256(block_content.encode("utf-8")).hexdigest()


class BaseBlockchainAdapter(ABC):
    """
    Abstract interface for blockchain storage adapters.
    Enables swapping local blockchain with a public testnet adapter in the future.
    """

    @abstractmethod
    def add_record(self, record_payload: Dict[str, Any]) -> str:
        """Adds a record payload and returns transaction/block record ID."""
        pass

    @abstractmethod
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a record payload by transaction/block record ID."""
        pass

    @abstractmethod
    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """Verifies overall blockchain cryptographic integrity."""
        pass

    @abstractmethod
    def verify_record(self, record_id: str, candidate_data: Dict[str, Any]) -> Tuple[str, float]:
        """Recomputes fingerprint and verifies on-chain record ('VERIFIED' or 'TAMPERED')."""
        pass


class LocalBlockchainService(BaseBlockchainAdapter):
    """
    Local SHA-256 linked blockchain implementation.
    """

    def __init__(self):
        self.chain: List[Block] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Creates the initial genesis block if chain is empty."""
        genesis_data = {
            "title": "Genesis Block",
            "message": "Face Identification Pipeline Blockchain Initialized",
            "fingerprint": compute_sha256_fingerprint("Genesis Block Initialized")
        }
        genesis_block = Block(
            index=0,
            timestamp=1700000000.0,
            data=genesis_data,
            previous_hash="0" * 64
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)

    def add_record(self, record_payload: Dict[str, Any]) -> str:
        """
        Appends a new record payload to the blockchain.
        Calculates SHA-256 fingerprint for payload if not explicitly present.
        
        Args:
            record_payload: Dictionary containing record metadata.
            
        Returns:
            The SHA-256 block hash representing the transaction / record ID.
        """
        payload_copy = dict(record_payload)
        if "fingerprint" not in payload_copy:
            payload_copy["fingerprint"] = compute_sha256_fingerprint(payload_copy)

        prev_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data=payload_copy,
            previous_hash=prev_block.hash
        )
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)
        return new_block.hash

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves record data by block hash or block index.
        
        Args:
            record_id: Block hash string or integer index string.
            
        Returns:
            Dictionary payload if found, None otherwise.
        """
        for block in self.chain:
            if block.hash == record_id or str(block.index) == str(record_id):
                return dict(block.data)
        return None

    def get_block(self, record_id: str) -> Optional[Block]:
        """Retrieves a Block instance by block hash."""
        for block in self.chain:
            if block.hash == record_id or str(block.index) == str(record_id):
                return block
        return None

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Validates cryptographic integrity of the entire blockchain.
        
        Returns:
            Tuple of (integrity_passed boolean, error_message string or None).
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # 1. Check current block points to previous block hash
            if current.previous_hash != previous.hash:
                return False, f"Block #{current.index} previous_hash does not match Block #{previous.index} hash. Chain broken!"

            # 2. Check current block hash matches calculated hash
            if current.hash != current.calculate_hash():
                return False, f"Block #{current.index} hash mismatch. Block data modified!"

        return True, None

    def verify_record(self, record_id: str, candidate_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Re-verifies candidate post data against the on-chain stored record.
        
        Args:
            record_id: Block hash string / record ID.
            candidate_data: Candidate metadata object to verify.
            
        Returns:
            Tuple of (status string 'VERIFIED'/'TAMPERED'/'RECORD_NOT_FOUND', confidence float).
        """
        stored_payload = self.get_record(record_id)
        if not stored_payload:
            return "RECORD_NOT_FOUND", 0.0

        # Compute fresh fingerprint of input candidate data
        new_fingerprint = compute_sha256_fingerprint(candidate_data)
        on_chain_fingerprint = stored_payload.get("fingerprint")

        if new_fingerprint == on_chain_fingerprint:
            return "VERIFIED", 1.0
        else:
            return "TAMPERED", 0.0

