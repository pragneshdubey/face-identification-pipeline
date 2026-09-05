"""
Full Pipeline Integration Script:
  FACE -> WEB SEARCH / CONSENT SEARCH -> CANDIDATES FOUND -> FACE VERIFICATION -> VERIFIED MATCH -> SHA-256 -> BLOCKCHAIN -> CHAIN INTEGRITY -> RE-VERIFICATION -> VERIFIED

Usage:
  python scripts/run_full_pipeline.py [--image path/to/image.jpg] [--threshold 0.5] [--provider {web,consent,mock}]
"""

import argparse
import datetime
import logging
import os
import sys

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from app.blockchain import LocalBlockchainService, compute_sha256_fingerprint
from app.face_engine import FaceIdentificationEngine, normalize_embedding
from app.search_engine import (
    CandidateFaceVerifier,
    ConsentRegistrySearchProvider,
    MockReverseImageSearchProvider,
    SerpApiGoogleLensProvider,
)


def run_pipeline(
    image_path: str,
    face_threshold: float = 0.5,
    provider_name: str = "web"
):
    print("==========================================================")
    print("  FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION PIPELINE  ")
    print("==========================================================")
    print(f"Query Image Path      : {image_path}")
    print(f"Face Similarity Cutoff: {face_threshold}")
    print(f"Selected Search Mode  : {provider_name.upper()}")

    # -----------------------------------------------------------------
    # STAGE 1: Face Detection & Embedding Generation
    # -----------------------------------------------------------------
    print("\n[FACE] Initializing InsightFace Engine & Extracting Embedding...")
    face_engine = FaceIdentificationEngine(threshold=face_threshold)
    query_emb = None

    if os.path.isfile(image_path):
        img, err = face_engine.load_image(image_path)
        if img is not None:
            query_faces = face_engine.extract_faces(img)
            if query_faces:
                query_emb = query_faces[0]["embedding"]
                det_score = query_faces[0].get("det_score", 1.0)
                bbox = query_faces[0]["bbox"]
                print(f"  -> Face detected! Count: {len(query_faces)} | BBox: {bbox} | Det Score: {det_score:.4f} | 512D Embedding generated.")

    if query_emb is None:
        if not os.path.isfile(image_path):
            print(f"[-] Error: Query image file not found: '{image_path}'")
            print("    Please provide a valid input face image path.")
            return
        else:
            print("[-] Error: No human faces detected in query image.")
            return

    # -----------------------------------------------------------------
    # STAGE 2: Search Stage (Web / Consent Registry / Mock)
    # -----------------------------------------------------------------
    p_lower = provider_name.lower()
    if p_lower in {"web", "serpapi"}:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            print("[-] Error: SERPAPI_API_KEY not set in environment or .env file.")
            print("    Please set SERPAPI_API_KEY or run with '--provider consent' / '--provider mock'.")
            return
        search_provider = SerpApiGoogleLensProvider(api_key=api_key)
        print("\n[WEB SEARCH] Executing Genuine Runtime Reverse Image Search (SerpApi Google Lens)...")
    elif p_lower == "mock":
        search_provider = MockReverseImageSearchProvider()
        print("\n[MOCK SEARCH] Executing Mock Reverse Image Search...")
    else:
        registry_file = os.path.join(PROJECT_ROOT, "data", "known_posts.json")
        search_provider = ConsentRegistrySearchProvider(registry_file=registry_file)
        print(f"\n[CONSENT-SCOPED SEARCH] Searching authorized post corpus '{registry_file}'...")

    search_response = search_provider.search_by_image(image_path)
    if not search_response.success:
        print(f"[-] Search Failed: {search_response.error_message}")
        return

    # -----------------------------------------------------------------
    # STAGE 3: Genuine Candidate Face Verification
    # -----------------------------------------------------------------
    print("\n[FACE VERIFICATION] Running Genuine Candidate Face Verification...")
    verifier = CandidateFaceVerifier(
        face_engine=face_engine,
        face_threshold=face_threshold,
        stop_on_first_match=True
    )

    results = verifier.verify_search_candidates(query_emb, search_response.candidates)
    verified_matches = [r for r in results if r.is_verified_match]

    if not verified_matches:
        print("\n[-] NO VERIFIED MATCH FOUND ABOVE THRESHOLD")
        print("    No on-chain record will be created.")
        return

    match = verified_matches[0]
    print(f"\n[VERIFIED MATCH]")
    print(f"  - Title          : {match.candidate_title}")
    print(f"  - URL            : {match.candidate_url}")
    print(f"  - Domain         : {match.source_domain}")
    print(f"  - Face Similarity: {match.face_similarity_score:.4f}")
    print(f"  - Candidate Image: {match.candidate_image_url}")

    # -----------------------------------------------------------------
    # STAGE 4: SHA-256 Fingerprinting
    # -----------------------------------------------------------------
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

    print(f"\n[SHA-256] Calculated Fingerprint:")
    print(f"  -> {fingerprint}")

    # -----------------------------------------------------------------
    # STAGE 5: Blockchain Upload & Record Storage
    # -----------------------------------------------------------------
    print("\n[BLOCKCHAIN] Initializing Local SHA-256 Linked Blockchain...")
    blockchain = LocalBlockchainService()

    tx_id = blockchain.add_record(canonical_post_data)
    print(f"\n[TRANSACTION/RECORD ID]")
    print(f"  -> {tx_id}")

    # -----------------------------------------------------------------
    # STAGE 6: Chain Integrity & On-Chain Re-verification
    # -----------------------------------------------------------------
    is_valid, err = blockchain.verify_chain()
    if is_valid:
        print(f"\n[CHAIN INTEGRITY] Cryptographic Hash Chain Validation: PASSED")
    else:
        print(f"\n[CHAIN INTEGRITY] Cryptographic Hash Chain Validation: FAILED -> {err}")
        return

    print("\n[RE-VERIFICATION] Retrieving Record & Re-evaluating SHA-256 Fingerprint...")
    verification_status, confidence = blockchain.verify_record(tx_id, canonical_post_data)

    print(f"\n[{verification_status}] On-Chain Tamper-Evident Record Verification:")
    print(f"  - Status    : {verification_status}")
    print(f"  - Confidence: {confidence * 100:.0f}%")
    print(f"  - Record ID : {tx_id}")
    print(f"  - Fingerprint: {fingerprint}")
    print("==========================================================")


if __name__ == "__main__":
    default_img = os.path.join(PROJECT_ROOT, "data", "test_images", "pragnesh_profile.jpg")
    if not os.path.isfile(default_img):
        default_img = os.path.join(PROJECT_ROOT, "data", "test_images", "consented_user.jpg")

    parser = argparse.ArgumentParser(description="Run Full Face Identification, Search & Blockchain Verification Pipeline.")
    parser.add_argument("--image", type=str, default=default_img, help="Path to input face image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Face similarity threshold (default: 0.5)")
    parser.add_argument("--provider", type=str, default="web", choices=["web", "consent", "mock", "serpapi"], help="Search provider (default: web)")
    args = parser.parse_args()

    run_pipeline(image_path=args.image, face_threshold=args.threshold, provider_name=args.provider)
