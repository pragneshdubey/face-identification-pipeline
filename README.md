# Face Identification & Web/Social Verification Pipeline

A modular, privacy-conscious Python pipeline for face detection, 512D embedding extraction, genuine runtime reverse-image search (SerpApi Google Lens), independent candidate face verification, and **tamper-evident local blockchain record storage**.

---

## Project Structure

```
face-identification-pipeline/
├── app/
│   ├── __init__.py           # Package exports
│   ├── face_engine.py        # Module A: Face Identification Engine (InsightFace)
│   ├── search_engine.py      # Module B & C: Web / Consent Search Providers & Candidate Face Verifier
│   └── blockchain.py         # Module D: Local SHA-256 Blockchain & Re-Verification
├── data/
│   ├── known_faces/          # Reference face images for identification
│   ├── test_images/          # Query test images
│   └── known_posts.json      # Controlled local registry of authorized posts / images
├── scripts/
│   ├── run_demo.py           # Face identification engine demo
│   ├── run_search_demo.py    # Search provider demo
│   ├── run_full_pipeline.py  # End-to-end face search & blockchain verification runner
│   └── verify_env.py         # Environment verification script
├── tests/
│   ├── test_face_engine.py   # Unit tests for Module A
│   ├── test_search_engine.py # Unit tests for Module B (Search Engine & Consent Registry)
│   ├── test_verification.py  # Unit tests for Module C (Candidate Face Verification)
│   └── test_blockchain.py    # Unit tests for Module D (Blockchain & Integrity)
├── .env.example              # Environment variables template
├── requirements.txt          # Project dependencies (pinned)
├── .gitignore                # Git ignore rules
└── README.md                 # Documentation
```

---

## End-to-End Pipeline Architecture

```
[ Input Query Face Image ] (Consenting Participant / User)
         │
         ▼
[ Face Identification Engine ] ────────> Extract 512D L2 Normalized Face Embedding via InsightFace
         │
         ▼
[ Genuine Runtime Web Search ] ─────────> Upload Image & Query Google Lens (SerpApi) / Local Registry
         │
         ▼
[ Candidate Face Verification ] ───────> Download Candidate Image -> InsightFace Detect Face -> Cosine Similarity
         │
         ▼ (VERIFIED MATCH)
[ SHA-256 Fingerprint Generator ] ─────> Compute Deterministic Hash of Discovered Match Payload
         │
         ▼
[ Local SHA-256 Blockchain ] ──────────> Store Fingerprint & Minimal Metadata in Linked Block Chain
         │
         ▼
[ On-Chain Re-Verification ] ──────────> Recompute Fingerprint & Verify Chain Cryptographic Integrity
```

---

## Technical Pipeline Overview

### 1. Face Identification (`app/face_engine.py`)
Detects human faces and extracts 512-dimensional L2 normalized feature vectors using InsightFace (`buffalo_l` pack on `CPUExecutionProvider`).

### 2. Genuine Runtime Web Search (`app/search_engine.py`)
Submits the query image at runtime via `SerpApiGoogleLensProvider`:
- Detects faces in the query image and automatically generates a face-focused crop (~25% margin) prior to Google Lens search for higher precision.
- Uploads the image or face crop to the SerpApi upload API (`https://serpapi.com/upload`) to obtain an `image_id`.
- Queries SerpApi Google Lens (`https://serpapi.com/search.json` with `engine="google_lens"`).
- Dynamically retrieves and parses visual candidate URLs, titles, source domains, and image URLs.
- *Never accepts a result solely because it came from the search engine.*

### 3. Candidate Face Verification (`CandidateFaceVerifier` in `app/search_engine.py`)
Performs independent visual verification on candidate images:
- Downloads each candidate image safely (with timeout, headers, and max size limits), falling back to Google's cached thumbnail URL if direct candidate URL retrieval fails (e.g. 403 or CDN decoding errors).
- Detects faces using InsightFace.
- Computes cosine similarity against the original query face embedding:
  $$\text{similarity} = \frac{\mathbf{v}_{\text{query}} \cdot \mathbf{v}_{\text{candidate}}}{\|\mathbf{v}_{\text{query}}\| \|\mathbf{v}_{\text{candidate}}\|}$$
- Confirms **`VERIFIED_MATCH`** only when detected face similarity score $\ge$ `face_threshold` (default `0.5`).
- *Facial similarity is used strictly as a biometric verification signal between image features and does not assert definitive legal identity.*

### 4. SHA-256 Fingerprinting (`app/blockchain.py`)
Computes a deterministic SHA-256 fingerprint hex string of the canonical discovered post payload.

### 5. Local SHA-256 Blockchain Storage (`app/blockchain.py`)
Stores the cryptographic fingerprint and minimal metadata in a local SHA-256 linked blockchain. Raw face images and unencrypted embeddings are **never** stored on-chain.

### 6. Blockchain Re-Verification (`app/blockchain.py`)
Recomputes the payload fingerprint and validates hash chain links (`verify_chain`). Returns **`VERIFIED`** if matching or **`TAMPERED`** if data was modified.

---

## Instructions & Provider Selection

### Provider Modes (`--provider`)
- `--provider web` *(Default)*: Performs genuine runtime reverse-image search via SerpApi Google Lens.
- `--provider consent`: Searches authorized post entries in `data/known_posts.json`.
- `--provider mock`: Offline provider for testing without API keys.

---

### Step 1: Setting Up Environment Key for Web Search
Add your SerpApi key to `.env`:
```env
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### Step 2: Running the End-to-End Pipeline
```bash
# Run genuine web search & verification (Default - Einstein demo image)
python scripts/run_full_pipeline.py --image data/test_images/einstein_demo.jpg --provider web --threshold 0.5

# Run consent-scoped registry search
python scripts/run_full_pipeline.py --image data/test_images/pragnesh_profile.jpg --provider consent

# Run offline mock search
python scripts/run_full_pipeline.py --image data/test_images/pragnesh_profile.jpg --provider mock
```

---

## Safety, Consent Scope & Limitations

- **Consent Scope**: For demonstration purposes, query images and candidates should be provided by consenting participants or authorized content. Unrestricted face tracking across arbitrary individuals on the public web is not supported or intended.
- **Provider Abstraction**: Interfacing via `BaseReverseImageSearchProvider` allows swapping web search with local registries or mock providers without changing pipeline verification logic.
- **Privacy Compliance**: Only non-sensitive metadata and cryptographic SHA-256 fingerprints are written to the blockchain.
- **Verification Signal Scope**: Facial embedding similarity is evaluated as a mathematical verification signal between detected face regions. It does not constitute proof of identity or make identity claims.

---

## Running the Complete Test Suite

```bash
python -m pytest -v
```

