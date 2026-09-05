# Face Identification & Web/Social Verification Pipeline

**Hacker House Goa 2026 — Task 3 Submission**

A full-stack, data-driven facial verification system that detects human faces, extracts 512-dimensional embeddings, executes real-time reverse image searches across the open web, independently verifies candidate face matches, and registers verified claims on a local cryptographic hash-chain blockchain for tamper-proof re-verification.

---

## System Workflow

```
Face Image
    ↓
Face Detection + 512D Embedding
    ↓
Face-Focused Reverse Image Search
    ↓
Web/Social Candidates
    ↓
Independent Face Verification
    ↓
Verified Candidate
    ↓
SHA-256 Fingerprint
    ↓
Local Hash-Chain Blockchain
    ↓
Blockchain Re-Verification
```

> **Note**: Search engine results are treated as unverified candidates. A web candidate is marked as a verified match only after independent facial embedding extraction and cosine similarity verification surpass the similarity threshold.

---

## Key Features

- **InsightFace Detection & 512D Embeddings**: Uses the `buffalo_l` ONNX model pack to detect face bounding boxes and extract L2-normalized 512-dimensional feature vectors.
- **Cosine Similarity Matching**: Computes mathematical vector similarity with configurable matching thresholds (default `0.50`).
- **Runtime Web & Social Search**: Integrates Google Lens via SerpApi (`SerpApiGoogleLensProvider`) to discover candidate web pages and social profiles at runtime.
- **Interactive Region Selection**: Supports optional manual face-region selection (drag & resize bounding box) alongside automatic face detection.
- **Independent Candidate Verification**: Automatically downloads candidate page images, crops face regions, and extracts embeddings to verify identity claims.
- **Resilient Image Ingestion**: Includes image size downscaling (1024px maximum dimension), candidate deduplication, request-level image caching, and fallback to thumbnail images.
- **Concurrent Pipeline Execution**: Accelerates candidate verification using bounded thread pool execution (`ThreadPoolExecutor`) with early termination upon finding the first verified match.
- **Cryptographic Fingerprinting**: Computes deterministic SHA-256 hashes over canonical post metadata (`source_url`, `domain`, `similarity_score`).
- **Local Linked Blockchain**: Records verified matches in a local SHA-256 hash-chain blockchain supporting previous-hash chaining and cryptographic tamper detection.
- **Blockchain Audit & Re-Verification**: Recomputes metadata fingerprints against stored ledger blocks to verify chain integrity and detect content tampering.
- **Full Stack Implementation**: Features a FastAPI Python backend, modern React 19 / Vite / Tailwind CSS frontend, CLI execution tools, and an automated pytest test suite.

---

## Hacker House Task 3 Requirement Mapping

| Requirement | Implementation Status | Description |
| :--- | :--- | :--- |
| **Face Scan / Identification** | **Passed** | InsightFace (`buffalo_l`) scans query images and detects face bounding boxes. |
| **Face Encoding** | **Passed** | Generates 512-dimensional L2-normalized embedding vectors for face comparison. |
| **Web/Social Search** | **Passed** | Executes Google Lens reverse image queries via SerpApi to discover open-web candidates. |
| **Genuine Runtime Search** | **Passed** | Performs live external HTTP search requests per query; no hardcoded demo identities. |
| **Candidate Verification** | **Passed** | Downloads candidate images and independently calculates facial similarity against query embeddings. |
| **Blockchain Upload** | **Passed** | Registers verified matches into a local SHA-256 linked hash-chain ledger. |
| **Blockchain Re-Verification** | **Passed** | Recomputes cryptographic fingerprints against local ledger history to audit integrity. |
| **Repository & Code Quality** | **Passed** | Clean repository structure with unit test coverage and frontend build scripts. |

---

## Architecture & Project Structure

```
face-identification-pipeline/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api.py           # FastAPI REST API endpoints (/api/health, /api/verify)
│   │   ├── face_engine.py   # InsightFace detection, 512D embeddings, cosine similarity
│   │   ├── search_engine.py # Google Lens (SerpApi), candidate verification, ThreadPoolExecutor
│   │   ├── blockchain.py    # Local SHA-256 linked hash-chain blockchain service
│   │   └── main.py          # Backend module entry point
│   ├── data/
│   │   ├── known_posts.json # Local consent registry test dataset
│   │   └── test_images/     # Sample query test images (e.g., einstein_demo.jpg)
│   ├── scripts/
│   │   ├── run_full_pipeline.py  # End-to-end CLI verification runner
│   │   ├── run_search_demo.py   # Reverse image search CLI runner
│   │   └── verify_env.py        # Environment diagnostic checker
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_api.py
│   │   ├── test_blockchain.py
│   │   ├── test_face_engine.py
│   │   ├── test_search_engine.py
│   │   └── test_verification.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # React UI components (FaceInputCard, RegionAdjuster, Stepper)
│   │   ├── main.tsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── .env.example
└── README.md
```

### Core File Responsibilities

- `backend/app/api.py`: Exposes REST endpoints (`/api/health`, `/api/verify`), handles file uploads, optional `crop_box` parameters, input validation, and pipeline coordination.
- `backend/app/face_engine.py`: Manages the thread-safe global `InsightFace` model singleton, face bounding box extraction, L2 normalization, and cosine similarity math.
- `backend/app/search_engine.py`: Executes SerpApi Google Lens reverse image queries, candidate ranking, deduplication, parallel candidate image downloads, and facial match evaluation.
- `backend/app/blockchain.py`: Implements the local SHA-256 hash-chain ledger, block creation, previous-hash linkage, chain integrity checks, and local ledger record re-verification.
- `frontend/src/App.tsx`: Renders the single-page web interface, face input dropzone, interactive crop region adjuster, real-time pipeline stepper, candidate verification details, and blockchain hash proof card.
- `backend/tests/`: Automated unit test suite validating API endpoints, security bounds, vector calculations, search providers, candidate verification, and local blockchain integrity.

---

## Technical Pipeline Detail

### 1. Face Detection & Embedding Extraction
- **Model**: InsightFace (`buffalo_l` model pack running on ONNX Runtime).
- **Execution**: The ONNX models are pre-warmed on server startup via FastAPI lifespan hooks to eliminate cold-start latency.
- **Output**: Generates a 512-dimensional floating-point vector normalized to unit L2 norm ($||v||_2 = 1.0$).

### 2. Reverse Image Search
- **Provider**: Google Lens via SerpApi (`SerpApiGoogleLensProvider`).
- **Execution**: Search queries are executed dynamically at runtime.
- **Cropping**: When a face is detected (or manually selected), the query image is cropped around the face region before uploading to Google Lens to focus the visual search on facial features rather than background elements.

### 3. Candidate Face Verification
- **Process**: Candidate page URLs returned by Google Lens are parsed, deduplicated, and ranked (prioritizing social media domains and high-quality image sources).
- **Concurrent Inspection**: Images are downloaded in parallel using a bounded `ThreadPoolExecutor` (4 workers) with strict network timeouts (2.0s connect, 3.0s read).
- **Facial Comparison**: Downloaded candidate images are passed to InsightFace to extract candidate face embeddings. Cosine similarity is computed against the query embedding:
  $$\text{Similarity}(q, c) = \frac{q \cdot c}{\|q\|_2 \|c\|_2}$$
- **Match Decision**: Candidates exceeding the similarity threshold (default `0.50`) are marked as verified matches.

> **Important**: Facial similarity is a mathematical verification signal and should not be interpreted as definitive proof of identity.

### 4. Blockchain Fingerprinting
- When a candidate passes facial verification, canonical post data (URL, domain, candidate title, similarity score) is formatted as a standardized JSON structure.
- A deterministic SHA-256 cryptographic fingerprint (64 hexadecimal characters) is generated over the canonical payload.

### 5. Local Hash-Chain Blockchain
- **Ledger Architecture**: Uses a local SHA-256 linked hash-chain blockchain.
- **Storage Scope**: Stores post metadata, timestamp, cryptographic fingerprint, block index, and the previous block's SHA-256 hash. Raw image files are not stored in the ledger.
- **Immutability**: Each block hash incorporates the `previous_hash`, ensuring that modifying any past record invalidates all subsequent block hashes.

### 6. Blockchain Re-Verification
- **Audit Procedure**: Given a record ID/block hash, the system retrieves the ledger entry, recomputes the fingerprint over the post data, and verifies previous-hash continuity across the chain.
- **Tamper Detection**: If any field in the recorded metadata is altered, fingerprint re-computation fails and reports a broken integrity state.

---

## Quick Start Guide

### Prerequisites
- **Python**: Version 3.11 or higher
- **Node.js**: Version 18 or higher (with `npm`)
- **Git**
- **SerpApi API Key**: Required for live Google Lens web search (optional if running in mock search mode)

### 1. Clone Repository

```bash
git clone https://github.com/pragneshdubey/face-identification-pipeline.git
cd face-identification-pipeline
```

### 2. Setup Python Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the repository root directory:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` to set your SerpApi API key:
```env
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### 5. Start Application (Two-Terminal Workflow)

#### Terminal 1 — Backend (FastAPI Server)
```powershell
# From repository root (with virtual environment activated)
python -m uvicorn backend.app.api:app --reload --port 8000
```
*Backend API will run at `http://127.0.0.1:8000`.*

#### Terminal 2 — Frontend (React / Vite Dev Server)
```powershell
# Navigate to frontend directory
cd frontend
npm install
npm run dev
```
*Frontend UI will run at `http://localhost:8443`.*

---

## System Verification

Check system health by sending a request to the backend health endpoint:

```bash
# PowerShell / Curl
curl http://127.0.0.1:8000/api/health
```

Expected JSON response:
```json
{
  "status": "online",
  "system": "FaceVerify HH Goa 2026",
  "face_engine": "InsightFace (buffalo_l)",
  "embedding_dimensions": 512,
  "default_threshold": 0.5,
  "serpapi_configured": true,
  "blockchain": "Local SHA-256 Hash Chain"
}
```

---

## Using the Web Interface

1. Open `http://localhost:8443` in your browser.
2. **Select Image**: Drag & drop a face image (JPG, PNG, WEBP) or click to browse. Alternatively, run with the default pre-loaded sample image (`einstein_demo.jpg`).
3. **Face Region Selection**:
   - By default, InsightFace automatically scans and detects faces across the full image.
   - Click **Adjust Face Region** to open the interactive selection tool. Drag and resize the bounding box around the target face and click **Use Selected Region**.
4. **Run Verification**: Click **Run Verification** to initiate the multi-stage pipeline:
   - **Face Detection**: Extracts bounding box and 512D embedding vector.
   - **Search**: Uploads face crop to Google Lens and retrieves open-web candidate links.
   - **Verify**: Downloads candidate images concurrently and verifies facial similarity against the query embedding.
   - **Blockchain**: Generates SHA-256 fingerprint and appends record to the local hash-chain.
   - **Re-Verify**: Audits local ledger record integrity and confirms cryptographic proof.
5. **Inspect Proofs**: Review the highest similarity match score, candidate source domain, SHA-256 fingerprint, block hash, and click **Copy** to copy full 64-character SHA-256 hashes to the clipboard.

---

## API Documentation

### `GET /api/health`
Returns system status, active configuration parameters, and SerpApi key presence.

### `POST /api/verify`
Executes the end-to-end verification pipeline.

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` *(UploadFile, optional)*: Image file upload (max 10 MB).
  - `image_name` *(string, optional)*: Filename of a pre-existing sample image in `data/test_images/`.
  - `provider` *(string, optional)*: Search provider selection — `"web"` (default, SerpApi Google Lens), `"mock"`, or `"consent"`.
  - `threshold` *(float, optional)*: Cosine similarity cutoff (default: `0.5`).
  - `crop_box` *(string, optional)*: Manual crop box coordinates formatted as `"xmin,ymin,xmax,ymax"`.

#### Sample Request (PowerShell)
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/verify" -Method Post -Form @{
    image_name = "einstein_demo.jpg"
    provider = "web"
    threshold = "0.5"
}
```

---

## Command Line Interface (CLI)

You can run the pipeline directly from the command line without launching the web frontend.

### Full Pipeline Integration Runner
```powershell
# Live Web Search Mode (SerpApi)
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/einstein_demo.jpg --provider web

# Offline Mock Mode
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/einstein_demo.jpg --provider mock

# Local Consent Registry Search Mode
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/einstein_demo.jpg --provider consent
```

### Search Demo Script
```powershell
python backend/scripts/run_search_demo.py --image backend/data/test_images/einstein_demo.jpg --provider web
```

---

## Automated Testing

Run the automated backend test suite using `pytest`:

```powershell
# From repository root
.venv\Scripts\python -m pytest -v backend/tests
```

### Test Suite Coverage
- **`test_api.py`**: API health endpoint, file upload size/extension validation, path traversal prevention, manual `crop_box` coordinate handling.
- **`test_face_engine.py`**: Image loading, vector math, L2 embedding normalization, cosine similarity calculations, threshold matching.
- **`test_search_engine.py`**: Candidate parsing, domain extraction, social media flag identification, candidate ranking, image downscaling, consent registry provider, mock search provider.
- **`test_verification.py`**: Candidate download error handling, low similarity rejection, early stopping behavior.
- **`test_blockchain.py`**: SHA-256 fingerprint generation, block creation, chain integrity audits, tamper detection.

*Current Development Status: 39 unit tests passing.*

---

## Production Frontend Build

To build the static production assets for the React frontend:

```powershell
cd frontend
npm run build
```

Compiled production bundle will be generated in `frontend/dist/`.

---

## Performance & Optimization

- **Model Pre-Warming**: InsightFace ONNX models are loaded into memory once on server startup.
- **Thread-Safe Shared Singleton**: `get_shared_face_analysis()` prevents duplicate model instantiations across requests.
- **Bounded Parallelism**: Candidate image verification runs concurrently using 4 thread pool workers.
- **Early Termination**: Halts remaining candidate inspections immediately upon finding the first verified match above threshold (`stop_on_first_match=True`).
- **Image Resizing**: Candidate images exceeding 1024px in width or height are downscaled before running ONNX inference.
- **Request-Level Caching**: Prevents redundant downloads of candidate images across duplicate URL references.

---

## Security & Privacy Considerations

- **API Credentials**: `SERPAPI_API_KEY` is managed server-side and never exposed to client-side code.
- **Upload Restrictions**: Uploaded files are strictly validated against allowed extensions (`.jpg`, `.jpeg`, `.png`, `.webp`) and a 10 MB maximum size limit.
- **Path Traversal Protection**: Query image names provided via API are sanitized and restricted to `backend/data/test_images/`.
- **Temp File Cleanup**: All temporary upload and cropped image files are removed in `finally:` cleanup blocks.
- **Blockchain Privacy**: The local ledger stores cryptographic hashes and post metadata only. Raw face images and biometric vectors are never written to the ledger.

---

## Responsible Use & Limitations

- **Intended Scope**: Designed for authorized content verification, controlled demonstrations, and consenting participants.
- **Search Dependency**: Live web search results depend on Google Lens indexing via SerpApi.
- **Environmental Factors**: Extreme lighting differences, heavy facial occlusion, or low resolution can affect cosine similarity scores.
- **Local Blockchain**: The local SHA-256 hash-chain ledger provides auditability for single-node demonstration environments and is not a decentralized public consensus network.

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Face Detection & Embeddings** | InsightFace (`buffalo_l`), ONNX Runtime, OpenCV |
| **Reverse Image Search** | Google Lens via SerpApi (`requests`) |
| **Backend Framework** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Blockchain Storage** | Local SHA-256 Linked Hash-Chain Ledger |
| **Frontend Framework** | React 19, TypeScript, Vite, Tailwind CSS |
| **Testing & Quality** | Pytest, TestClient, Oxlint |
