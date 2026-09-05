"""
Demo script to execute Reverse Image Search using Web or Consent Registry providers.

Usage:
  python scripts/run_search_demo.py [--image path/to/image.jpg] [--provider {web,consent,mock}]
"""

import argparse
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from app.search_engine import (
    CandidateVerificationService,
    ConsentRegistrySearchProvider,
    MockReverseImageSearchProvider,
    SerpApiGoogleLensProvider,
)


def run_search_demo(image_path: str, provider_name: str = "web"):
    print("==========================================================")
    print("        REVERSE IMAGE SEARCH DEMO (SERPAPI / REGISTRY)   ")
    print("==========================================================")

    p_lower = provider_name.lower()
    if p_lower in {"web", "serpapi"}:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            print("[-] Error: SERPAPI_API_KEY not set in environment or .env file.")
            return
        provider = SerpApiGoogleLensProvider(api_key=api_key)
        print("[+] Using Provider: SerpApi Google Lens Provider (Genuine Web Search)")
    elif p_lower == "mock":
        provider = MockReverseImageSearchProvider()
        print("[+] Using Provider: Mock Reverse Image Search Provider")
    else:
        registry_file = os.path.join(PROJECT_ROOT, "data", "known_posts.json")
        provider = ConsentRegistrySearchProvider(registry_file=registry_file)
        print(f"[+] Using Provider: Consent Registry Search Provider ('{registry_file}')")

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

    print(f"--- All Extracted Candidates ({len(ranked)}) ---")
    for idx, cand in enumerate(ranked[:5], start=1):
        social_flag = "[SOCIAL MEDIA]" if cand.is_social_media else "[WEB PAGE]"
        title = cand.title.encode('ascii', errors='replace').decode('ascii')
        print(f"  [{idx}] {social_flag} {title}")
        print(f"      URL   : {cand.url}")
        print(f"      Domain: {cand.source_domain} | Relevance Score: {cand.relevance_score}")

    print("\n==========================================================")
    print("Search Demo Finished.")


if __name__ == "__main__":
    default_img = os.path.join(PROJECT_ROOT, "data", "test_images", "pragnesh_profile.jpg")
    if not os.path.isfile(default_img):
        default_img = os.path.join(PROJECT_ROOT, "data", "test_images", "consented_user.jpg")

    parser = argparse.ArgumentParser(description="Run Reverse Image Search Demo.")
    parser.add_argument("--image", type=str, default=default_img, help="Path to query image")
    parser.add_argument("--provider", type=str, default="web", choices=["web", "consent", "mock", "serpapi"], help="Search provider (default: web)")
    args = parser.parse_args()

    run_search_demo(image_path=args.image, provider_name=args.provider)
