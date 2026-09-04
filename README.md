# Face Identification Pipeline

A modular Python pipeline for face detection, 512D embedding extraction, identity matching, reverse image search (SerpApi Google Lens), candidate face verification, and **tamper-evident local blockchain record storage**.

---

## Project Structure

```
face-identification-pipeline/
├── app/
│   ├── __init__.py           # Package exports
│   ├── face_engine.py        # Module A: Face Identification Engine
│   ├── search_engine.py      # Module B & C: Reverse Image Search & Candidate Face Verification
│   └── blockchain.py         # Module D: Local SHA-256 Blockchain & Verification
├── data/
│   ├── known_faces/          # Reference face images for registration
│   └── test_images/          # Query test images
├── scripts/
│   ├── run_demo.py           # Face identification demo
│   ├── run_search_demo.py    # Search candidate extraction demo
│   ├── run_full_pipeline.py  # End-to-end face search & blockchain verification runner
│   └── verify_env.py         # Environment verification script
├── tests/
│   ├── test_face_engine.py   # Unit tests for Module A
│   ├── test_search_engine.py # Unit tests for Module B
│   ├── test_verification.py  # Unit tests for Module C (Face Verification)
│   └── test_blockchain.py    # Unit tests for Module D (Blockchain)
├── .env.example              # Environment variables template
├── requirements.txt          # Project dependencies (pinned)
├── .gitignore                # Git ignore rules
└── README.md                 # Documentation
```

---

## End-to-End Pipeline Architecture

```
[ Input Face Image ]
         │
         ▼
[ Face Identification Engine ] ────> Extract 512D L2 Normalized Face Embedding
         │
         ▼
[ Genuine SerpApi Google Lens ] ───> Retrieve Reverse Image Search Candidates
         │
         ▼
[ Candidate Face Verification ] ───> Download Candidate Image -> InsightFace Detect Face -> Cosine Similarity
         │
         ▼ (VERIFIED MATCH)
[ SHA-256 Fingerprint Generator ] ─> Compute Deterministic Hash of Discovered Match Payload
         │
         ▼
[ Local SHA-256 Blockchain ] ──────> Store Fingerprint & Minimal Metadata in Linked Block Chain
         │
         ▼
[ On-Chain Re-Verification ] ──────> Recompute Fingerprint & Verify Chain Cryptographic Integrity
```

---

## Blockchain Architecture (`app/blockchain.py`)

### 1. Why a Local Blockchain is Used
- **Zero Cost & Reliability**: Provides a fast, deterministic, local SHA-256 linked chain without requiring paid external web3 infrastructure or testnet faucet tokens during development.
- **Privacy Compliance**: Raw personal images or 512D embeddings are **never** written to the chain. Only the cryptographic SHA-256 fingerprint and minimal metadata are recorded.

### 2. Pluggable Testnet Adapter Architecture
The codebase defines an abstract base interface `BaseBlockchainAdapter`:

```python
class BaseBlockchainAdapter(ABC):
    def add_record(self, record_payload: Dict[str, Any]) -> str: ...
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]: ...
    def verify_chain(self) -> Tuple[bool, Optional[str]]: ...
    def verify_record(self, record_id: str, candidate_data: Dict[str, Any]) -> Tuple[str, float]: ...
```

To migrate to a public testnet (e.g. Ethereum Sepolia, Polygon Amoy, or Solana Devnet) in the future:
1. Create `PublicTestnetBlockchainAdapter(BaseBlockchainAdapter)`.
2. Wrap smart contract invocations (`addRecord`, `verifyRecord`) inside the adapter.
3. Pass `PublicTestnetBlockchainAdapter` into `run_full_pipeline.py` without modifying any other stage of the pipeline.

---

### 3. Data Stored On-Chain
Only minimal, non-sensitive metadata is stored:

```json
{
  "title": "Discovered Match Title",
  "source_url": "https://www.linkedin.com/in/target",
  "source_domain": "linkedin.com",
  "candidate_image_url": "https://media.licdn.com/dms/image/sample.jpg",
  "face_similarity_score": 0.8542,
  "verification_status": "VERIFIED_MATCH",
  "timestamp": "2026-09-04T00:43:00.000Z",
  "fingerprint": "a3f890b21c4e976f... (64-char SHA-256 hex string)"
}
```

---

### 4. SHA-256 Fingerprinting & Tamper Verification
- **SHA-256 Fingerprint**: Calculated using deterministic canonical JSON formatting (`sort_keys=True, separators=(',', ':')`).
- **Chain Linking**: Each block stores `hash = SHA256(index + timestamp + previous_hash + payload)`. Modifying any field in an earlier block invalidates `previous_hash` references across all downstream blocks.
- **Re-Verification**: To verify a record, the candidate data payload is re-hashed. If the recomputed fingerprint matches the stored on-chain fingerprint, status returns **`VERIFIED`**. If any property (title, URL, score, or timestamp) was modified, status returns **`TAMPERED`**.

---

## Running the Complete Test Suite

```bash
python -m unittest discover -s tests
```

---

## Running End-to-End Pipeline

```bash
# Live search mode (with SERPAPI_API_KEY in .env)
python scripts/run_full_pipeline.py --image "data/test_images/Akshay_Kumar_National_Award_for_Padman_(cropped).jpg.webp" --threshold 0.5

# Offline mock mode
python scripts/run_full_pipeline.py --use-mock
```
