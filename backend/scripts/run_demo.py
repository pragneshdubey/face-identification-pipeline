"""
Demo script to test the Face Identification Pipeline.

Usage:
  python scripts/run_demo.py [--threshold 0.5]
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.face_engine import FaceIdentificationEngine


def run_demo(threshold: float = 0.5):
    known_faces_dir = os.path.join(PROJECT_ROOT, "data", "known_faces")
    test_images_dir = os.path.join(PROJECT_ROOT, "data", "test_images")

    print("==========================================================")
    print("      FACE IDENTIFICATION PIPELINE - DEMO RUNNER         ")
    print("==========================================================")
    print(f"Known Faces Directory : {known_faces_dir}")
    print(f"Test Images Directory : {test_images_dir}")
    print(f"Matching Threshold    : {threshold}")
    print("----------------------------------------------------------")

    engine = FaceIdentificationEngine(threshold=threshold)

    # 1. Load known face database
    print("\n[1] Loading known faces database...")
    load_results = engine.load_known_faces_directory(known_faces_dir)

    if not load_results:
        print("  -> No image files (.jpg, .png, etc.) found in 'data/known_faces/'.")
        print("  -> To register a person, place their image (e.g., 'alice.jpg') in 'data/known_faces/'.")
    else:
        for person_id, success in load_results.items():
            status = "REGISTERED" if success else "FAILED"
            print(f"  -> {person_id}: {status}")

    # 2. Process test images
    print("\n[2] Processing test images...")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    test_files = []
    if os.path.isdir(test_images_dir):
        test_files = [
            f for f in os.listdir(test_images_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

    if not test_files:
        print("  -> No test image files found in 'data/test_images/'.")
        print("  -> To run identification tests, place test images in 'data/test_images/'.")
        return

    for test_file in test_files:
        test_path = os.path.join(test_images_dir, test_file)
        print(f"\nEvaluating: '{test_file}'")
        results = engine.identify_faces(test_path)

        for idx, res in enumerate(results, start=1):
            if res.is_match:
                print(f"  Face #{idx}: MATCHED -> Identity: {res.person_id} | Similarity: {res.similarity_score:.4f} | BBox: {res.bbox}")
            elif res.error_message:
                print(f"  Face #{idx}: ERROR -> {res.error_message}")
            else:
                print(f"  Face #{idx}: NO MATCH -> Best Similarity: {res.similarity_score:.4f} (Below threshold {threshold}) | BBox: {res.bbox}")

    print("\n==========================================================")
    print("Demo Execution Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Identification Pipeline Demo.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold (default: 0.5)")
    args = parser.parse_args()
    run_demo(threshold=args.threshold)

