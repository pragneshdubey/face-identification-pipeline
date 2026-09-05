"""
Main entry point for the Face Identification Pipeline application.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.face_engine import FaceIdentificationEngine


def main():
    print("==================================================")
    print("     Face Identification Pipeline Engine          ")
    print("==================================================")

    known_faces_dir = os.path.join(PROJECT_ROOT, "data", "known_faces")
    test_images_dir = os.path.join(PROJECT_ROOT, "data", "test_images")

    print(f"[+] Initializing Face Identification Engine...")
    engine = FaceIdentificationEngine(threshold=0.5)

    print(f"[+] Scanning known faces directory: '{known_faces_dir}'")
    loaded = engine.load_known_faces_directory(known_faces_dir)
    print(f"[+] Registered identities: {len(loaded)}")

    if not loaded:
        print("[-] Note: No known identity images found in 'data/known_faces/'.")
        print("    Add target person photos (e.g. 'alice.jpg') to 'data/known_faces/' to register identities.")

    print("\n[+] Identification Engine initialized and ready.")


if __name__ == "__main__":
    main()
