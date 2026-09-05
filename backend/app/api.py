"""
FastAPI Backend Layer for Face Identification & Blockchain Verification Pipeline.

Exposes REST API endpoints for frontend UI integration:
  - GET  /api/health : System status & config health check
  - POST /api/verify : Run end-to-end verification pipeline on an uploaded/selected face image
"""

import datetime
import os
import sys
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import numpy as np
from dotenv import load_dotenv

# Ensure backend root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load environment variables from backend/.env or root .env
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(REPO_ROOT, ".env"))

from app.blockchain import LocalBlockchainService, compute_sha256_fingerprint
from app.face_engine import FaceIdentificationEngine, compute_cosine_similarity
from app.search_engine import (
    CandidateFaceVerifier,
    ConsentRegistrySearchProvider,
    MockReverseImageSearchProvider,
    SerpApiGoogleLensProvider,
)

app = FastAPI(
    title="Face Verification & Blockchain API",
    description="Genuine Runtime Reverse Image Search & Local Blockchain Verification API",
    version="1.0.0"
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8443",
        "http://127.0.0.1:8443"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def get_health():
    """
    Returns system status and active pipeline configuration parameters.
    """
    serp_key = os.environ.get("SERPAPI_API_KEY")
    return {
        "status": "online",
        "system": "FaceVerify HH Goa 2026",
        "face_engine": "InsightFace (buffalo_l)",
        "embedding_dimensions": 512,
        "default_threshold": 0.5,
        "serpapi_configured": bool(serp_key),
        "blockchain": "Local SHA-256 Hash Chain",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }


@app.post("/api/verify")
async def verify_image(
    file: Optional[UploadFile] = File(None),
    image_name: Optional[str] = Form(None),
    provider: str = Form("web"),
    threshold: float = Form(0.5)
):
    """
    Executes the complete face verification & blockchain pipeline.
    Accepts an uploaded image file or a predefined sample image name.
    """
    temp_path = None
    try:
        # Determine image source
        if file is not None:
            filename = os.path.basename(file.filename or "uploaded_image.jpg")
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )

            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file size ({len(contents)} bytes) exceeds maximum limit of 10 MB."
                )

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(contents)
                temp_path = tmp.name
            target_image_path = temp_path
        elif image_name:
            filename = os.path.basename(image_name)
            test_images_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "data", "test_images"))
            target_image_path = os.path.abspath(os.path.join(test_images_dir, filename))

            # Ensure target_image_path is strictly inside test_images_dir and exists
            if not target_image_path.startswith(test_images_dir) or not os.path.isfile(target_image_path):
                raise HTTPException(status_code=400, detail=f"Invalid or unauthorized test image name '{filename}'.")
        else:
            # Default fallback to Einstein demo image
            default_demo = os.path.join(PROJECT_ROOT, "data", "test_images", "einstein_demo.jpg")
            if os.path.isfile(default_demo):
                target_image_path = default_demo
                filename = "einstein_demo.jpg"
            else:
                raise HTTPException(status_code=400, detail="No image provided and default demo image missing.")


        # Stage 1: Face Detection & Embedding
        face_engine = FaceIdentificationEngine(threshold=threshold)
        img, err = face_engine.load_image(target_image_path)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {err}")

        query_faces = face_engine.extract_faces(img)
        if not query_faces:
            return {
                "success": False,
                "face_detected": False,
                "error": "No human face detected in the query image.",
                "filename": filename,
                "threshold": threshold,
                "candidates_evaluated": 0,
                "match_found": False,
                "logs": [
                    {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Loading input image ({filename})...", "status": "info"},
                    {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "No face detected in query image.", "status": "warning"}
                ]
            }

        query_emb = query_faces[0]["embedding"]
        det_score = float(query_faces[0].get("det_score", 1.0))
        bbox = [int(v) for v in query_faces[0]["bbox"]]

        logs = [
            {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Loading input image ({filename})...", "status": "info"},
            {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Face detected — confidence {det_score:.4f}", "status": "success"},
            {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Generating 512D L2-normalized face embedding...", "status": "info"}
        ]

        # Stage 2: Search Provider Selection
        p_lower = provider.lower()
        if p_lower in {"web", "serpapi"}:
            api_key = os.environ.get("SERPAPI_API_KEY")
            if not api_key:
                # Fallback to mock if API key missing
                search_provider = MockReverseImageSearchProvider()
                logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "SERPAPI_API_KEY missing. Falling back to Mock search provider.", "status": "warning"})
            else:
                search_provider = SerpApiGoogleLensProvider(api_key=api_key)
                logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Uploading face crop to Google Lens (SerpApi)...", "status": "info"})
        elif p_lower == "mock":
            search_provider = MockReverseImageSearchProvider()
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Executing Mock reverse image search...", "status": "info"})
        else:
            registry_file = os.path.join(PROJECT_ROOT, "data", "known_posts.json")
            search_provider = ConsentRegistrySearchProvider(registry_file=registry_file)
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Searching consent registry '{registry_file}'...", "status": "info"})

        search_response = search_provider.search_by_image(target_image_path)
        if not search_response.success:
            return {
                "success": False,
                "face_detected": True,
                "detection_score": round(det_score, 4),
                "embedding_generated": True,
                "error": f"Search failed: {search_response.error_message}",
                "filename": filename,
                "threshold": threshold,
                "candidates_evaluated": 0,
                "match_found": False,
                "logs": logs + [{"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Search failed: {search_response.error_message}", "status": "warning"}]
            }

        candidates = search_response.candidates
        total_retrieved = len(candidates)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"{total_retrieved} candidates retrieved from open-web search", "status": "success"})
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Running genuine candidate face verification...", "status": "info"})

        # Stage 3: Candidate Face Verification
        verifier = CandidateFaceVerifier(
            face_engine=face_engine,
            face_threshold=threshold,
            stop_on_first_match=True
        )

        results = verifier.verify_search_candidates(query_emb, candidates)
        verified_matches = [r for r in results if r.is_verified_match]

        highest_sim = 0.0
        for r in results:
            if r.face_similarity_score > highest_sim:
                highest_sim = r.face_similarity_score

        if not verified_matches:
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Highest candidate similarity score: {highest_sim:.4f}", "status": "warning"})
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Verification threshold: {threshold:.4f}", "status": "info"})
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "No candidate exceeded threshold — no match found", "status": "warning"})
            logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Blockchain record not created — verification not passed", "status": "warning"})

            return {
                "success": True,
                "face_detected": True,
                "detection_score": round(det_score, 4),
                "bbox": bbox,
                "embedding_generated": True,
                "filename": filename,
                "candidates_evaluated": total_retrieved,
                "match_found": False,
                "highest_similarity": round(highest_sim, 4),
                "threshold": threshold,
                "candidate": None,
                "blockchain": None,
                "logs": logs
            }

        match = verified_matches[0]
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Best candidate similarity score: {match.face_similarity_score:.4f}", "status": "success"})
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Verification threshold: {threshold:.4f}", "status": "info"})
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Candidate VERIFIED — exceeds threshold", "status": "success"})

        # Stage 4: SHA-256 Fingerprinting
        canonical_post_data = {
            "title": match.candidate_title,
            "source_url": match.candidate_url,
            "source_domain": match.source_domain,
            "candidate_image_url": match.candidate_image_url,
            "face_similarity_score": round(match.face_similarity_score, 4),
            "verification_status": match.verification_status,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        fingerprint = compute_sha256_fingerprint(canonical_post_data)
        canonical_post_data["fingerprint"] = fingerprint
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "SHA-256 fingerprint generated", "status": "success"})

        # Stage 5 & 6: Blockchain Storage & Re-verification
        blockchain = LocalBlockchainService()
        tx_id = blockchain.add_record(canonical_post_data)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Blockchain record created", "status": "success"})

        is_valid, chain_err = blockchain.verify_chain()
        chain_status = "PASSED" if is_valid else f"FAILED ({chain_err})"
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Chain integrity check {chain_status}", "status": "success" if is_valid else "warning"})

        rever_status, confidence = blockchain.verify_record(tx_id, canonical_post_data)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"On-chain re-verification {rever_status} ({confidence * 100:.0f}%)", "status": "success"})

        return {
            "success": True,
            "face_detected": True,
            "detection_score": round(det_score, 4),
            "bbox": bbox,
            "embedding_generated": True,
            "filename": filename,
            "candidates_evaluated": total_retrieved,
            "match_found": True,
            "threshold": threshold,
            "candidate": {
                "title": match.candidate_title,
                "source_url": match.candidate_url,
                "domain": match.source_domain,
                "candidate_image_url": match.candidate_image_url,
                "similarity": round(match.face_similarity_score, 4)
            },
            "blockchain": {
                "fingerprint": fingerprint,
                "record_id": tx_id,
                "chain_integrity": chain_status,
                "reverification": rever_status,
                "confidence": round(confidence * 100, 1)
            },
            "logs": logs
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)

