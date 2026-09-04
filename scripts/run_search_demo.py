"""
Demo script to execute Web / Social Media Reverse Image Search using SerpApi Google Lens.

Usage:
  python scripts/run_search_demo.py [--image path/to/image.jpg] [--use-mock]
"""

import argparse
import os
import sys

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.search_engine import (
    CandidateVerificationService,
    MockReverseImageSearchProvider,
    SerpApiGoogleLensProvider,
)


def run_search_demo(image_path: str, use_mock: bool = False):
    print("==========================================================")
    print("      REVERSE IMAGE SEARCH DEMO (SERPAPI GOOGLE LENS)    ")
    print("==========================================================")

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key and not use_mock:
        print("[!] Warning: SERPAPI_API_KEY is not set in environment or .env file.")
        print("    Falling back to Mock Reverse Image Search Provider.")
        use_mock = True

    if use_mock:
        provider = MockReverseImageSearchProvider()
        print("[+] Using Provider: Mock Reverse Image Search Provider")
    else:
        provider = SerpApiGoogleLensProvider(api_key=api_key)
        print("[+] Using Provider: SerpApi Google Lens Provider")

    print(f"[+] Query Image: '{image_path}'")
    print("----------------------------------------------------------")

    print("[+] Executing reverse image search...")
    response = provider.search_by_image(image_path)

    if not response.success:
        print(f"[-] Search Failed: {response.error_message}")
        print(f"    Log Info: {response.log_info}")
        return

    print(f"[+] Search Successful! Total candidates returned: {response.total_results}")
    print(f"    Log Info: {response.log_info}")

    # Rank candidates
    ranked = CandidateVerificationService.rank_candidates(response.candidates)
    social_candidates = CandidateVerificationService.filter_social_media_candidates(ranked)

    print(f"\n--- Social Media Match Candidates ({len(social_candidates)}) ---")
    if not social_candidates:
        print("  -> No direct social media matches found.")
    else:
        for idx, cand in enumerate(social_candidates, start=1):
            title = cand.title.encode('ascii', errors='replace').decode('ascii')
            print(f"  [{idx}] Title        : {title}")
            print(f"      URL          : {cand.url}")
            print(f"      Domain       : {cand.source_domain}")
            print(f"      Score        : {cand.relevance_score}")
            print(f"      Thumbnail    : {cand.thumbnail_url or 'N/A'}")
            print()

    print(f"--- All Extracted Web Candidates ({len(ranked)}) ---")
    for idx, cand in enumerate(ranked[:5], start=1):
        social_flag = "[SOCIAL MEDIA]" if cand.is_social_media else "[WEB PAGE]"
        title = cand.title.encode('ascii', errors='replace').decode('ascii')
        print(f"  [{idx}] {social_flag} {title}")
        print(f"      URL   : {cand.url}")
        print(f"      Domain: {cand.source_domain} | Relevance Score: {cand.relevance_score}")

    print("\n==========================================================")
    print("Search Demo Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Reverse Image Search Demo.")
    parser.add_argument("--image", type=str, default=os.path.join(PROJECT_ROOT, "data", "test_images", "Akshay_Kumar_National_Award_for_Padman_(cropped).jpg.webp"), help="Path to query image")
    parser.add_argument("--use-mock", action="store_true", help="Force mock search provider")
    args = parser.parse_args()

    run_search_demo(image_path=args.image, use_mock=args.use_mock)
