# Face Identification & Web/Social Verification Pipeline

A modular, privacy-conscious full-stack application featuring a React/Vite UI frontend and a Python FastAPI backend pipeline for face detection, 512D embedding extraction, genuine runtime reverse-image search (SerpApi Google Lens), independent candidate face verification, and **tamper-evident local blockchain record storage**.

---

## Project Architecture & Structure

```
face-identification-pipeline/
├── frontend/                     # React 19 + Vite 8 + Tailwind CSS 4 UI
│   ├── src/
│   │   ├── App.tsx               # Main UI Component & API Integration
│   │   ├── main.tsx              # Application entry point
│   │   └── index.css             # Tailwind styling & custom utilities
│   ├── package.json              # Frontend dependencies & scripts
│   └── vite.config.ts            # Vite build configuration with API proxy (/api)
│
├── backend/                      # Python Verification & Blockchain Engine
│   ├── app/
│   │   ├── __init__.py           # Package exports
│   │   ├── api.py                # FastAPI HTTP REST API endpoints (/api/verify, /api/health)
│   │   ├── face_engine.py        # Module A: InsightFace 512D Face Detection & Embeddings
│   │   ├── search_engine.py      # Module B & C: Web Search (SerpApi Google Lens) & Candidate Face Verifier
│   │   └── blockchain.py         # Module D: Local SHA-256 Hash-Chain Blockchain & Re-Verification
│   ├── data/
│   │   ├── test_images/          # Query test images (e.g. einstein_demo.jpg)
│   │   └── known_posts.json      # Authorized consent registry
│   ├── scripts/
│   │   ├── run_full_pipeline.py  # End-to-end CLI pipeline runner
│   │   ├── run_search_demo.py    # Standalone search provider runner
│   │   └── run_demo.py           # Standalone face engine runner
│   ├── tests/
│   │   ├── conftest.py           # Pytest path resolution configuration
│   │   ├── test_api.py           # Unit tests for FastAPI REST API
│   │   ├── test_blockchain.py    # Unit tests for Blockchain Ledger & Re-verification
│   │   ├── test_face_engine.py   # Unit tests for InsightFace Engine & Vector Math
│   │   ├── test_search_engine.py # Unit tests for Web Search & Candidate Ranking
│   │   └── test_verification.py  # Unit tests for Candidate Visual Verification
│   └── requirements.txt          # Backend Python dependencies
│
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

---

## Technical Pipeline Overview

### 1. Face Identification (`backend/app/face_engine.py`)
Detects human faces and extracts 512-dimensional L2 normalized feature vectors using InsightFace (`buffalo_l` pack on `CPUExecutionProvider`).

### 2. Genuine Runtime Web Search (`backend/app/search_engine.py`)
Submits the query image at runtime via `SerpApiGoogleLensProvider`:
- Detects faces in the query image and automatically generates a face-focused crop (~25% margin) prior to Google Lens search for higher precision.
- Uploads the image or face crop to the SerpApi upload API (`https://serpapi.com/upload`) to obtain an `image_id`.
- Queries SerpApi Google Lens (`https://serpapi.com/search.json` with `engine="google_lens"`).
- Dynamically retrieves and parses visual candidate URLs, titles, source domains, and image URLs.
- *Never accepts a result solely because it came from the search engine.*

### 3. Candidate Face Verification (`CandidateFaceVerifier` in `backend/app/search_engine.py`)
Performs independent visual verification on candidate images:
- Downloads each candidate image safely (with timeout, headers, and max size limits), falling back to Google's cached thumbnail URL if direct candidate URL retrieval fails (e.g. 403 or CDN decoding errors).
- Detects faces using InsightFace.
- Computes L2-normalized cosine similarity against the original query face embedding:
  $$\text{similarity} = \frac{\mathbf{v}_{\text{query}} \cdot \mathbf{v}_{\text{candidate}}}{\|\mathbf{v}_{\text{query}}\| \|\mathbf{v}_{\text{candidate}}\|}$$
- Confirms **`VERIFIED_MATCH`** only when detected face similarity score $\ge$ `face_threshold` (default `0.5`).
- *Facial similarity is used strictly as a biometric verification signal between image features and does not assert definitive legal identity.*

### 4. SHA-256 Fingerprinting (`backend/app/blockchain.py`)
Computes a deterministic SHA-256 fingerprint hex string of the canonical discovered post payload.

### 5. Local SHA-256 Blockchain Storage (`backend/app/blockchain.py`)
Stores the cryptographic fingerprint and minimal metadata in a local SHA-256 linked blockchain. Raw face images and unencrypted embeddings are **never** stored on-chain.

### 6. Blockchain Re-Verification (`backend/app/blockchain.py`)
Recomputes the payload fingerprint and validates hash chain links (`verify_chain`). Returns **`VERIFIED`** if matching or **`TAMPERED`** if data was modified.

---

## Instructions: Running the Backend & Frontend

### 1. Environment Setup
Add your SerpApi key to `.env` in the repository root:
```env
SERPAPI_API_KEY=your_serpapi_api_key_here
```

### 2. Starting the Backend API Server
```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Launch FastAPI server with Uvicorn
python -m uvicorn backend.app.api:app --host 127.0.0.1 --port 8000 --reload
```
The API server will run on `http://127.0.0.1:8000`. Endpoint health check is available at `http://127.0.0.1:8000/api/health`.

### 3. Starting the Frontend UI
```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
```
The React UI will open on `http://localhost:8443` (or Vite's configured dev port) and proxy `/api` requests to the Python backend.

---

## API Documentation

### `POST /api/verify`
Executes the full verification pipeline on an uploaded or selected face image.
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  * `file`: (Optional) Uploaded image file (`JPG`/`PNG`)
  * `image_name`: (Optional) Predefined sample image name (default: `einstein_demo.jpg`)
  * `provider`: Search provider mode (`web`, `consent`, `mock`)
  * `threshold`: Face similarity cutoff (default: `0.5`)

### `GET /api/health`
Returns active system health status, configuration details, and SerpApi key presence.

---

## Running CLI & Tests

### Running the CLI Pipeline
```bash
# Run genuine web search & verification (Einstein demo image)
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/einstein_demo.jpg --provider web --threshold 0.5

# Run consent-scoped registry search
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/pragnesh_profile.jpg --provider consent

# Run offline mock search
python backend/scripts/run_full_pipeline.py --image backend/data/test_images/einstein_demo.jpg --provider mock
```

### Running Backend Unit Tests
```bash
python -m pytest -v backend/tests
```

### Building Frontend
```bash
cd frontend
npm run build
```

---

## Safety, Consent Scope & Limitations

- **Consent Scope**: For demonstration purposes, query images and candidates should be provided by consenting participants or authorized content. Unrestricted face tracking across arbitrary individuals on the public web is not supported or intended.
- **Provider Abstraction**: Interfacing via `BaseReverseImageSearchProvider` allows swapping web search with local registries or mock providers without changing pipeline verification logic.
- **API Key Security**: `SERPAPI_API_KEY` is handled strictly server-side by the Python backend and is **never** exposed to the React frontend or client browsers.
- **Privacy Compliance**: Only non-sensitive metadata and cryptographic SHA-256 fingerprints are written to the blockchain.
- **Verification Signal Scope**: Facial embedding similarity is evaluated as a mathematical verification signal between detected face regions. It does not constitute proof of identity or make identity claims.
