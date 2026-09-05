"""
FastAPI Backend Layer for Face Identification & Blockchain Verification Pipeline.

Exposes REST API endpoints for frontend UI integration:
  - GET  /api/health : System status & config health check
  - POST /api/verify : Run end-to-end verification pipeline on an uploaded/selected face image
"""

from contextlib import asynccontextmanager
import datetime
import os
import sys
import tempfile
import time
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
from app.face_engine import FaceIdentificationEngine, compute_cosine_similarity, get_shared_face_analysis
from app.search_engine import (
    CandidateFaceVerifier,
    ConsentRegistrySearchProvider,
    MockReverseImageSearchProvider,
    SerpApiGoogleLensProvider,
)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Pre-warm InsightFace model on server startup so requests run in sub-second time
    try:
        t0 = time.time()
        print("[Lifespan] Pre-warming InsightFace model...")
        get_shared_face_analysis()
        print(f"[Lifespan] InsightFace model pre-warmed in {time.time() - t0:.3f}s")
    except Exception as e:
        print(f"[Lifespan Error] Failed to pre-warm InsightFace model: {e}")
    yield


app = FastAPI(
    title="Face Verification & Blockchain API",
    description="Genuine Runtime Reverse Image Search & Local Blockchain Verification API",
    version="1.0.0",
    lifespan=lifespan
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
    threshold: float = Form(0.5),
    crop_box: Optional[str] = Form(None)
):
    """
    Executes the complete face verification & blockchain pipeline.
    Accepts an uploaded image file or a predefined sample image name.
    Optionally accepts crop_box="xmin,ymin,xmax,ymax" for manual face region selection.
    """
    temp_path = None
    extra_temp_paths = []
    t_pipeline_start = time.time()
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
        t_load_start = time.time()
        face_engine = FaceIdentificationEngine(threshold=threshold)
        img, err = face_engine.load_image(target_image_path)
        t_load_end = time.time()
        print(f"[Performance] Image load time: {t_load_end - t_load_start:.4f}s")
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {err}")

        # Handle optional manual crop region
        crop_offset_x = 0
        crop_offset_y = 0
        if crop_box:
            try:
                import json
                if crop_box.startswith("[") and crop_box.endswith("]"):
                    coords = json.loads(crop_box)
                else:
                    coords = [int(v.strip()) for v in crop_box.split(",") if v.strip()]
                if len(coords) == 4:
                    c_xmin, c_ymin, c_xmax, c_ymax = [int(v) for v in coords]
                    img_h, img_w = img.shape[:2]

                    c_xmin = max(0, min(c_xmin, img_w - 1))
                    c_ymin = max(0, min(c_ymin, img_h - 1))
                    c_xmax = max(c_xmin + 1, min(c_xmax, img_w))
                    c_ymax = max(c_ymin + 1, min(c_ymax, img_h))

                    if (c_xmax - c_xmin) >= 10 and (c_ymax - c_ymin) >= 10:
                        cropped_img = img[c_ymin:c_ymax, c_xmin:c_xmax]
                        crop_offset_x = c_xmin
                        crop_offset_y = c_ymin

                        crop_ext = os.path.splitext(target_image_path)[1] or ".jpg"
                        with tempfile.NamedTemporaryFile(suffix=f"_crop{crop_ext}", delete=False) as tmp_crop:
                            tmp_crop_path = tmp_crop.name
                        cv2.imwrite(tmp_crop_path, cropped_img)
                        if temp_path:
                            extra_temp_paths.append(temp_path)
                        temp_path = tmp_crop_path
                        target_image_path = tmp_crop_path
                        img = cropped_img
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Selected crop_box dimensions ({c_xmax - c_xmin}x{c_ymax - c_ymin}) are too small."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="crop_box parameter must contain exactly 4 integer coordinates: xmin,ymin,xmax,ymax."
                    )
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(status_code=400, detail=f"Invalid crop_box format: {e}")

        t_det_start = time.time()
        query_faces = face_engine.extract_faces(img)
        t_det_end = time.time()
        print(f"[Performance] Face detection & embedding extraction time: {t_det_end - t_det_start:.4f}s")

        if not query_faces:
            err_msg = "No human face detected in the selected region." if crop_box else "No human face detected in the query image."
            return {
                "success": False,
                "face_detected": False,
                "error": err_msg,
                "filename": filename,
                "threshold": threshold,
                "candidates_evaluated": 0,
                "match_found": False,
                "logs": [
                    {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Loading input image ({filename})...", "status": "info"},
                    {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": err_msg, "status": "warning"}
                ]
            }

        query_emb = query_faces[0]["embedding"]
        det_score = float(query_faces[0].get("det_score", 1.0))
        bbox = [
            crop_offset_x + int(query_faces[0]["bbox"][0]),
            crop_offset_y + int(query_faces[0]["bbox"][1]),
            crop_offset_x + int(query_faces[0]["bbox"][2]),
            crop_offset_y + int(query_faces[0]["bbox"][3])
        ]

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

        t_search_start = time.time()
        search_response = search_provider.search_by_image(target_image_path)
        t_search_end = time.time()
        print(f"[Performance] Open web search time: {t_search_end - t_search_start:.4f}s")

        if not search_response.success:
            raw_err = search_response.error_message or "Open-web search failed"
            if "Read timed out" in raw_err or "timeout" in raw_err.lower():
                clean_err = "Google Lens search timed out before candidates could be retrieved."
            else:
                clean_err = raw_err

            return {
                "success": False,
                "face_detected": True,
                "detection_score": round(det_score, 4),
                "bbox": bbox,
                "embedding_generated": True,
                "search_status": "failed",
                "error_stage": "web_search",
                "error": f"Search failed: {clean_err}",
                "filename": filename,
                "threshold": threshold,
                "candidates_evaluated": 0,
                "match_found": False,
                "logs": logs + [{"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Search failed: {clean_err}", "status": "warning"}]
            }

        candidates = search_response.candidates
        total_retrieved = len(candidates)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"{total_retrieved} candidates retrieved from open-web search", "status": "success"})
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Running genuine candidate face verification...", "status": "info"})

        # Stage 3: Candidate Face Verification
        t_verif_start = time.time()
        verifier = CandidateFaceVerifier(
            face_engine=face_engine,
            face_threshold=threshold,
            stop_on_first_match=True
        )

        results = verifier.verify_search_candidates(query_emb, candidates)
        t_verif_end = time.time()
        print(f"[Performance] Candidate face verification time: {t_verif_end - t_verif_start:.4f}s")

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

            print(f"[Performance] Total pipeline time: {time.time() - t_pipeline_start:.4f}s")
            return {
                "success": True,
                "face_detected": True,
                "detection_score": round(det_score, 4),
                "bbox": bbox,
                "embedding_generated": True,
                "search_status": "success",
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
        t_bc_start = time.time()
        blockchain = LocalBlockchainService()
        tx_id = blockchain.add_record(canonical_post_data)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": "Blockchain record created", "status": "success"})

        is_valid, chain_err = blockchain.verify_chain()
        chain_status = "PASSED" if is_valid else f"FAILED ({chain_err})"
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"Chain integrity check {chain_status}", "status": "success" if is_valid else "warning"})

        rever_status, confidence = blockchain.verify_record(tx_id, canonical_post_data)
        logs.append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": f"On-chain re-verification {rever_status} ({confidence * 100:.0f}%)", "status": "success"})
        t_bc_end = time.time()
        print(f"[Performance] Blockchain operations time: {t_bc_end - t_bc_start:.4f}s")
        print(f"[Performance] Total pipeline time: {time.time() - t_pipeline_start:.4f}s")

        return {
            "success": True,
            "face_detected": True,
            "detection_score": round(det_score, 4),
            "bbox": bbox,
            "embedding_generated": True,
            "search_status": "success",
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
                "block_hash": tx_id,
                "previous_hash": blockchain.chain[-1].previous_hash if (blockchain.chain and len(blockchain.chain) > 0) else "0000000000000000000000000000000000000000000000000000000000000000",
                "chain_integrity": chain_status,
                "reverification": rever_status,
                "confidence": round(confidence * 100, 1)
            },
            "logs": logs
        }
    finally:
        for tp in ([temp_path] + extra_temp_paths):
            if tp and os.path.exists(tp):
                try:
                    os.remove(tp)
                except Exception:
                    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)

