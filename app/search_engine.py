"""
Web and Social Media Search Engine Module for Reverse Image Search and Consent-Scoped Search.

Provides provider abstraction, ConsentRegistrySearchProvider, SerpApi Google Lens integration,
and candidate verification service for identifying face image matches.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from urllib.parse import urlparse
import cv2
import numpy as np
import requests

from app.face_engine import FaceIdentificationEngine, compute_cosine_similarity

logger = logging.getLogger(__name__)

# Standard social media & public profile domains for filtering
SOCIAL_MEDIA_DOMAINS: Set[str] = {
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "pinterest.com",
    "tiktok.com",
    "github.com",
    "youtube.com",
    "reddit.com",
    "flickr.com",
    "tumblr.com",
    "medium.com",
}


@dataclass
class SearchResultCandidate:
    """Dataclass representing a candidate result from reverse image search."""
    title: str
    url: str
    source_domain: str
    thumbnail_url: Optional[str] = None
    original_image_url: Optional[str] = None
    is_social_media: bool = False
    relevance_score: float = 0.0
    raw_metadata: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class SearchResponse:
    """Standardized response from a reverse image search provider."""
    success: bool
    query_image: str
    provider_name: str
    candidates: List[SearchResultCandidate] = field(default_factory=list)
    total_results: int = 0
    error_message: Optional[str] = None
    log_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateVerificationResult:
    """Dataclass representing the face verification outcome for a search candidate."""
    candidate_url: str
    candidate_title: str
    source_domain: str
    candidate_image_url: Optional[str] = None
    face_similarity_score: float = 0.0
    is_verified_match: bool = False
    verification_status: str = "PENDING"  # VERIFIED_MATCH, REJECTED_LOW_SIMILARITY, REJECTED_NO_FACE, REJECTED_DOWNLOAD_ERROR, REJECTED_NO_IMAGE_URL
    rejection_reason: Optional[str] = None
    best_face_bbox: Optional[Tuple[int, int, int, int]] = None
    raw_candidate: Optional[SearchResultCandidate] = field(default=None, repr=False)


def extract_domain(url: str) -> str:
    """
    Extracts the root domain name from a URL.
    
    Args:
        url: Web URL string.
        
    Returns:
        Cleaned domain string (e.g. 'instagram.com').
    """
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_social_media_domain(domain: str) -> bool:
    """
    Checks whether a domain belongs to a known social media platform.
    
    Args:
        domain: Cleaned domain string.
        
    Returns:
        True if social media domain, False otherwise.
    """
    if not domain:
        return False
    domain_lower = domain.lower()
    for social in SOCIAL_MEDIA_DOMAINS:
        if domain_lower == social or domain_lower.endswith("." + social):
            return True
    return False


def optimize_image_for_upload(image_path: str, max_bytes: int = 500 * 1024) -> Tuple[bytes, str]:
    """
    Reads an image file and compresses/resizes it if it exceeds SerpApi max size (500 KB).
    
    Args:
        image_path: Local path to the image file.
        max_bytes: Maximum allowed file size in bytes (default 500 KB).
        
    Returns:
        Tuple of (image bytes, mime_type string).
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    file_size = os.path.getsize(image_path)
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"

    with open(image_path, "rb") as f:
        raw_bytes = f.read()

    if file_size <= max_bytes and ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return raw_bytes, mime_type

    img = cv2.imread(image_path)
    if img is None:
        return raw_bytes, mime_type

    quality = 90
    encode_ext = ".jpg"
    mime_type = "image/jpeg"

    while quality >= 30:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded_img = cv2.imencode(encode_ext, img, encode_param)
        if success:
            encoded_bytes = encoded_img.tobytes()
            if len(encoded_bytes) <= max_bytes:
                return encoded_bytes, mime_type
        quality -= 15

    scale = 0.75
    while scale > 0.1:
        new_w = max(100, int(img.shape[1] * scale))
        new_h = max(100, int(img.shape[0] * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        success, encoded_img = cv2.imencode(encode_ext, resized, encode_param)
        if success:
            encoded_bytes = encoded_img.tobytes()
            if len(encoded_bytes) <= max_bytes:
                return encoded_bytes, mime_type
        scale -= 0.2

    return raw_bytes, mime_type


class BaseReverseImageSearchProvider(ABC):
    """
    Abstract Base Class for Reverse Image Search providers.
    """

    @abstractmethod
    def search_by_image(self, image_input: str) -> SearchResponse:
        """
        Performs reverse image search for a given image file path or URL.
        
        Args:
            image_input: Local file path or public image URL.
            
        Returns:
            SearchResponse object containing candidate results and metadata.
        """
        pass


class ConsentRegistrySearchProvider(BaseReverseImageSearchProvider):
    """
    Default Search Provider searching only a controlled registry of posts/images
    explicitly authorized by consenting participants (data/known_posts.json).
    """

    REQUIRED_FIELDS = {"id", "url", "image_url", "platform", "owner"}

    def __init__(
        self,
        registry_file: Optional[str] = None,
        posts_data: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initializes the Consent Registry Search Provider.
        
        Args:
            registry_file: Path to known_posts.json file.
            posts_data: Explicit list of post dicts (for testing).
        """
        if registry_file is None and posts_data is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            registry_file = os.path.join(project_root, "data", "known_posts.json")

        self.registry_file = registry_file
        self._explicit_posts = posts_data

    def load_posts(self) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Loads and validates consented post entries from registry file or explicit list.
        
        Returns:
            Tuple of (list of valid post dicts, error message or None).
        """
        if self._explicit_posts is not None:
            posts = self._explicit_posts
        else:
            if not self.registry_file or not os.path.isfile(self.registry_file):
                return [], f"Consent registry file not found: '{self.registry_file}'"

            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                posts = data.get("posts", [])
            except Exception as e:
                return [], f"Failed to parse consent registry file '{self.registry_file}': {str(e)}"

        valid_posts = []
        for idx, post in enumerate(posts):
            if not isinstance(post, dict):
                continue
            missing = self.REQUIRED_FIELDS - set(post.keys())
            if missing:
                logger.warning(f"[ConsentRegistry] Post index #{idx} missing required fields {missing}. Skipping.")
                continue
            valid_posts.append(post)

        return valid_posts, None

    def search_by_image(self, image_input: str) -> SearchResponse:
        """
        Queries the authorized consent registry for search candidates.
        
        Args:
            image_input: Input query image file path or URL.
            
        Returns:
            SearchResponse object containing registered candidates.
        """
        provider_name = "Consent Registry Search Provider"
        log_info = {
            "registry_file": self.registry_file,
            "query_input": image_input
        }

        logger.info(f"[CONSENT-SCOPED SEARCH] Query Input: '{image_input}' using {provider_name}")

        posts, err = self.load_posts()
        if err:
            logger.error(f"[ConsentRegistry] {err}")
            return SearchResponse(
                success=False,
                query_image=image_input,
                provider_name=provider_name,
                error_message=err,
                log_info=log_info
            )

        candidates: List[SearchResultCandidate] = []
        for post in posts:
            domain = extract_domain(post["url"]) or f"{post['platform'].lower()}.com"
            title = f"Authorized Post by {post['owner']} on {post['platform']}"

            candidate = SearchResultCandidate(
                title=title,
                url=post["url"],
                source_domain=domain,
                thumbnail_url=post["image_url"],
                original_image_url=post["image_url"],
                is_social_media=is_social_media_domain(domain) or True,
                raw_metadata=post
            )
            candidates.append(candidate)

        log_info["total_candidates"] = len(candidates)
        logger.info(f"[CANDIDATES FOUND] Total consent-scoped candidates retrieved: {len(candidates)}")

        return SearchResponse(
            success=True,
            query_image=image_input,
            provider_name=provider_name,
            candidates=candidates,
            total_results=len(candidates),
            log_info=log_info
        )


class SerpApiGoogleLensProvider(BaseReverseImageSearchProvider):
    """
    Optional Reverse Image Search Provider using SerpApi Google Lens Engine.
    """

    SEARCH_ENDPOINT = "https://serpapi.com/search.json"
    UPLOAD_ENDPOINT = "https://serpapi.com/image"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes SerpApi provider.
        
        Args:
            api_key: SerpApi API Key string. Defaults to SERPAPI_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("SERPAPI_API_KEY")

    def _create_face_crop_if_possible(self, image_path: str) -> Tuple[str, bool]:
        """
        Detects faces in query image using InsightFace. If a face is detected, crops it
        with ~25% margin and saves to a temporary image file for face-focused Lens search.
        
        Args:
            image_path: Local path to query image.
            
        Returns:
            Tuple of (path_to_use, is_cropped_boolean).
        """
        try:
            img, err = FaceIdentificationEngine.load_image(image_path)
            if img is None:
                return image_path, False

            face_engine = FaceIdentificationEngine()
            faces = face_engine.extract_faces(img)
            if not faces or "bbox" not in faces[0]:
                logger.info("[FACE CROP] No face detected in query image. Using full image for Lens search.")
                return image_path, False

            bbox = faces[0]["bbox"]  # (left, top, right, bottom)
            x1, y1, x2, y2 = bbox
            h, w, _ = img.shape

            margin_x = int((x2 - x1) * 0.25)
            margin_y = int((y2 - y1) * 0.25)

            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)

            face_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            temp_crop_dir = os.path.join(os.path.dirname(image_path), ".face_crops")
            os.makedirs(temp_crop_dir, exist_ok=True)
            crop_filename = f"crop_{os.path.basename(image_path)}"
            crop_path = os.path.join(temp_crop_dir, crop_filename)
            
            cv2.imwrite(crop_path, face_crop)
            logger.info(f"[FACE CROP] Created face crop region ({crop_x2-crop_x1}x{crop_y2-crop_y1}) for face-focused Google Lens search.")
            return crop_path, True
        except Exception as e:
            logger.warning(f"[FACE CROP] Failed to create face crop: {e}. Falling back to full image.")
            return image_path, False

    def _upload_local_image(self, image_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Uploads a local image file to SerpApi Image API to obtain an image_id.
        
        Args:
            image_path: Path to local image file.
            
        Returns:
            Tuple of (image_id string or None, error message or None).
        """
        try:
            img_bytes, mime_type = optimize_image_for_upload(image_path)
        except Exception as e:
            return None, f"Failed to optimize/read image: {str(e)}"

        files = {
            "image": ("image.jpg", img_bytes, mime_type)
        }
        params = {
            "api_key": self.api_key
        }

        logger.info(f"[SerpApi] Uploading image '{image_path}' ({len(img_bytes)} bytes) to SerpApi upload endpoint ({self.UPLOAD_ENDPOINT})...")

        try:
            resp = requests.post(self.UPLOAD_ENDPOINT, params=params, files=files, timeout=30)
            if resp.status_code != 200:
                return None, f"SerpApi image upload failed (HTTP {resp.status_code}): {resp.text}"

            data = resp.json()
            image_id = data.get("image_id")
            if not image_id:
                return None, f"SerpApi upload response missing 'image_id': {data}"

            logger.info(f"[SerpApi] Image upload successful. Received image_id: '{image_id}'")
            return image_id, None
        except Exception as e:
            return None, f"Exception during SerpApi image upload: {str(e)}"

    def search_by_image(self, image_input: str) -> SearchResponse:
        """
        Performs reverse image search using SerpApi Google Lens engine.
        
        Args:
            image_input: Local file path or public image URL.
            
        Returns:
            SearchResponse object with candidates.
        """
        provider_name = "SerpApi (Google Lens)"
        log_info = {
            "endpoint": self.SEARCH_ENDPOINT,
            "engine": "google_lens",
            "query_input": image_input
        }

        logger.info(f"[OPEN-WEB SEARCH] Query Input: '{image_input}' using {provider_name}")

        if not self.api_key:
            err_msg = "SERPAPI_API_KEY is not set. Please set the SERPAPI_API_KEY environment variable or pass api_key."
            logger.error(f"[SerpApi] {err_msg}")
            return SearchResponse(
                success=False,
                query_image=image_input,
                provider_name=provider_name,
                error_message=err_msg,
                log_info=log_info
            )

        params: Dict[str, Any] = {
            "api_key": self.api_key,
            "engine": "google_lens",
            "hl": "en"
        }

        is_url = image_input.startswith("http://") or image_input.startswith("https://")

        if is_url:
            params["url"] = image_input
            log_info["search_type"] = "public_url"
        else:
            if not os.path.isfile(image_input):
                return SearchResponse(
                    success=False,
                    query_image=image_input,
                    provider_name=provider_name,
                    error_message=f"Local image file not found: '{image_input}'",
                    log_info=log_info
                )

            upload_path, is_cropped = self._create_face_crop_if_possible(image_input)
            image_id, upload_err = self._upload_local_image(upload_path)
            if not image_id:
                return SearchResponse(
                    success=False,
                    query_image=image_input,
                    provider_name=provider_name,
                    error_message=upload_err,
                    log_info=log_info
                )
            params["image_id"] = image_id
            log_info["search_type"] = "face_crop_upload" if is_cropped else "image_id_upload"
            log_info["image_id"] = image_id
            log_info["is_face_crop"] = is_cropped

        try:
            resp = requests.get(self.SEARCH_ENDPOINT, params=params, timeout=30)
            if resp.status_code != 200:
                err_msg = f"SerpApi Google Lens query failed (HTTP {resp.status_code}): {resp.text}"
                logger.error(f"[SerpApi] {err_msg}")
                return SearchResponse(
                    success=False,
                    query_image=image_input,
                    provider_name=provider_name,
                    error_message=err_msg,
                    log_info=log_info
                )

            data = resp.json()
            if "error" in data:
                return SearchResponse(
                    success=False,
                    query_image=image_input,
                    provider_name=provider_name,
                    error_message=f"SerpApi returned error: {data['error']}",
                    log_info=log_info
                )

            candidates: List[SearchResultCandidate] = []
            visual_matches = data.get("visual_matches", [])
            exact_matches = data.get("exact_matches", [])
            all_matches = exact_matches + visual_matches

            for match in all_matches:
                title = match.get("title") or match.get("source") or "Untitled Match"
                link = match.get("link") or match.get("source_url") or ""
                thumbnail = match.get("thumbnail")
                original = match.get("original") or match.get("image")
                domain = extract_domain(link)
                is_social = is_social_media_domain(domain)

                if link:
                    candidates.append(SearchResultCandidate(
                        title=title,
                        url=link,
                        source_domain=domain,
                        thumbnail_url=thumbnail,
                        original_image_url=original,
                        is_social_media=is_social,
                        raw_metadata=match
                    ))

            log_info["total_visual_matches"] = len(visual_matches)
            log_info["total_exact_matches"] = len(exact_matches)
            logger.info(f"[CANDIDATES FOUND] Total open-web candidates retrieved: {len(candidates)}")

            return SearchResponse(
                success=True,
                query_image=image_input,
                provider_name=provider_name,
                candidates=candidates,
                total_results=len(candidates),
                log_info=log_info
            )

        except requests.Timeout:
            err_msg = "SerpApi request timed out after 30 seconds."
            logger.error(f"[SerpApi] {err_msg}")
            return SearchResponse(
                success=False,
                query_image=image_input,
                provider_name=provider_name,
                error_message=err_msg,
                log_info=log_info
            )
        except Exception as e:
            err_msg = f"Unexpected exception during SerpApi search: {str(e)}"
            logger.error(f"[SerpApi] {err_msg}")
            return SearchResponse(
                success=False,
                query_image=image_input,
                provider_name=provider_name,
                error_message=err_msg,
                log_info=log_info
            )


class MockReverseImageSearchProvider(BaseReverseImageSearchProvider):
    """
    Offline Mock Reverse Image Search Provider for testing and validation.
    """

    def __init__(self, mock_candidates: Optional[List[SearchResultCandidate]] = None):
        self.mock_candidates = mock_candidates

    def search_by_image(self, image_input: str) -> SearchResponse:
        provider_name = "Mock Search Provider"
        logger.info(f"[SEARCH STARTED] Query Input: '{image_input}' using {provider_name}")

        if self.mock_candidates is not None:
            candidates = self.mock_candidates
        else:
            candidates = [
                SearchResultCandidate(
                    title="Profile Photo on LinkedIn",
                    url="https://www.linkedin.com/in/sample-profile",
                    source_domain="linkedin.com",
                    thumbnail_url="https://example.com/thumb1.jpg",
                    is_social_media=True,
                    relevance_score=0.9
                ),
                SearchResultCandidate(
                    title="Public Instagram Post",
                    url="https://www.instagram.com/p/sample123",
                    source_domain="instagram.com",
                    thumbnail_url="https://example.com/thumb2.jpg",
                    is_social_media=True,
                    relevance_score=0.85
                ),
                SearchResultCandidate(
                    title="News Article Image",
                    url="https://www.example-news.com/tech-article",
                    source_domain="example-news.com",
                    thumbnail_url="https://example.com/thumb3.jpg",
                    is_social_media=False,
                    relevance_score=0.6
                )
            ]

        logger.info(f"[CANDIDATES FOUND] Total candidates retrieved: {len(candidates)}")
        return SearchResponse(
            success=True,
            query_image=image_input,
            provider_name=provider_name,
            candidates=candidates,
            total_results=len(candidates),
            log_info={"mock": True}
        )


class CandidateVerificationService:
    """
    Service for filtering, ranking, and selecting reverse image search candidates.
    """

    @staticmethod
    def filter_social_media_candidates(candidates: List[SearchResultCandidate]) -> List[SearchResultCandidate]:
        """
        Filters candidates to return only those originating from social media platforms.
        
        Args:
            candidates: List of SearchResultCandidate instances.
            
        Returns:
            Filtered list of candidates with is_social_media == True.
        """
        return [c for c in candidates if c.is_social_media]

    @staticmethod
    def rank_candidates(candidates: List[SearchResultCandidate]) -> List[SearchResultCandidate]:
        """
        Ranks candidates based on domain authority, metadata completeness, and social media priority.
        
        Args:
            candidates: List of SearchResultCandidate instances.
            
        Returns:
            Ranked list of candidates sorted by calculated relevance score descending.
        """
        for c in candidates:
            score = 0.5  # Base score
            if c.is_social_media:
                score += 0.3
            if c.thumbnail_url or c.original_image_url:
                score += 0.1
            if c.title and c.title != "Untitled Match":
                score += 0.1
            c.relevance_score = round(min(1.0, score), 2)

        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)


class CandidateFaceVerifier:
    """
    Genuine Face Verification Engine for Search Candidates.
    Downloads/reads candidate images, detects faces via InsightFace, and verifies face embeddings
    against the query face embedding using cosine similarity.
    """

    def __init__(
        self,
        face_engine: Optional[FaceIdentificationEngine] = None,
        face_threshold: float = 0.5,
        download_timeout: float = 10.0,
        max_image_bytes: int = 10 * 1024 * 1024,
        stop_on_first_match: bool = True
    ):
        """
        Initializes the candidate face verifier.
        
        Args:
            face_engine: FaceIdentificationEngine instance (lazy-loaded if None).
            face_threshold: Minimum cosine similarity score for a face match (default: 0.5).
            download_timeout: Timeout in seconds for downloading candidate images (default: 10.0s).
            max_image_bytes: Max size in bytes for candidate image downloads (default: 10 MB).
            stop_on_first_match: Whether to stop verification after finding first verified match.
        """
        self.face_engine = face_engine or FaceIdentificationEngine(threshold=face_threshold)
        self.face_threshold = face_threshold
        self.download_timeout = download_timeout
        self.max_image_bytes = max_image_bytes
        self.stop_on_first_match = stop_on_first_match

    def download_candidate_image(self, image_url: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Downloads or reads a candidate image safely using OpenCV or requests.
        Supports both local file paths and remote HTTP URLs.
        
        Args:
            image_url: Public image URL or local file path.
            
        Returns:
            Tuple of (OpenCV BGR image numpy array or None, error message string or None).
        """
        if not image_url:
            return None, "No image URL provided."

        # Check if candidate image URL is a local file path
        if os.path.isfile(image_url):
            img = cv2.imread(image_url)
            if img is None:
                return None, f"Failed to load local candidate image file '{image_url}'"
            return img, None

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            resp = requests.get(image_url, headers=headers, timeout=self.download_timeout, stream=True)
            if resp.status_code != 200:
                return None, f"HTTP Error {resp.status_code} while downloading image from '{image_url}'"

            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_image_bytes:
                return None, f"Image size ({content_length} bytes) exceeds limit of {self.max_image_bytes} bytes."

            image_bytes = bytearray()
            for chunk in resp.iter_content(chunk_size=65536):
                image_bytes.extend(chunk)
                if len(image_bytes) > self.max_image_bytes:
                    return None, f"Image stream size exceeded maximum limit of {self.max_image_bytes} bytes."

            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return None, f"OpenCV failed to decode image downloaded from '{image_url}'."

            return img, None
        except requests.Timeout:
            return None, f"Download timed out ({self.download_timeout}s) for URL '{image_url}'."
        except Exception as e:
            return None, f"Failed to download candidate image from '{image_url}': {str(e)}"

    def verify_candidate(
        self,
        query_embedding: np.ndarray,
        candidate: SearchResultCandidate
    ) -> CandidateVerificationResult:
        """
        Verifies a single search result candidate against the query face embedding.
        
        Args:
            query_embedding: L2 normalized face embedding vector of the original query image.
            candidate: SearchResultCandidate object.
            
        Returns:
            CandidateVerificationResult object.
        """
        title_clean = candidate.title.encode('ascii', errors='ignore').decode('ascii') if candidate.title else ""
        logger.info(f"[VERIFYING CANDIDATE] '{title_clean}' | URL: {candidate.url}")

        primary_url = candidate.original_image_url
        fallback_url = candidate.thumbnail_url
        target_url = primary_url or fallback_url

        if not target_url:
            reason = "No candidate image URL available for visual face verification."
            logger.info(f"[REJECTED] {reason}")
            return CandidateVerificationResult(
                candidate_url=candidate.url,
                candidate_title=candidate.title,
                source_domain=candidate.source_domain,
                candidate_image_url=None,
                verification_status="REJECTED_NO_IMAGE_URL",
                rejection_reason=reason,
                raw_candidate=candidate
            )

        img, err = self.download_candidate_image(target_url)
        used_url = target_url

        # If primary image download failed and cached thumbnail fallback is available, try fallback
        if img is None and primary_url and fallback_url and primary_url != fallback_url:
            logger.info(f"[IMAGE FALLBACK] Primary candidate image download failed ({err}). Trying cached thumbnail URL...")
            img, fallback_err = self.download_candidate_image(fallback_url)
            if img is not None:
                used_url = fallback_url
                err = None
            else:
                err = f"Primary failed: {err} | Fallback failed: {fallback_err}"

        image_url = used_url

        if img is None:
            reason = f"Candidate image download failed: {err}"
            logger.info(f"[REJECTED] {reason}")
            return CandidateVerificationResult(
                candidate_url=candidate.url,
                candidate_title=candidate.title,
                source_domain=candidate.source_domain,
                candidate_image_url=used_url,
                verification_status="REJECTED_DOWNLOAD_ERROR",
                rejection_reason=reason,
                raw_candidate=candidate
            )

        # Detect faces in candidate image
        faces = self.face_engine.extract_faces(img)
        num_faces = len(faces)
        logger.info(f"[FACE DETECTED] Found {num_faces} face(s) in candidate image.")

        if num_faces == 0:
            reason = "No faces detected in the downloaded candidate image."
            logger.info(f"[REJECTED] {reason}")
            return CandidateVerificationResult(
                candidate_url=candidate.url,
                candidate_title=candidate.title,
                source_domain=candidate.source_domain,
                candidate_image_url=image_url,
                verification_status="REJECTED_NO_FACE",
                rejection_reason=reason,
                raw_candidate=candidate
            )

        best_score = -1.0
        best_bbox = None

        for face in faces:
            score = compute_cosine_similarity(query_embedding, face["embedding"])
            if score > best_score:
                best_score = score
                best_bbox = face["bbox"]

        best_score = max(0.0, float(best_score))
        logger.info(f"[SIMILARITY SCORE] Candidate Best Face Similarity: {best_score:.4f} (Threshold: {self.face_threshold:.4f})")

        if best_score >= self.face_threshold:
            logger.info(f"[VERIFIED MATCH] Candidate '{candidate.title}' matched query face with score {best_score:.4f} >= {self.face_threshold:.4f}!")
            return CandidateVerificationResult(
                candidate_url=candidate.url,
                candidate_title=candidate.title,
                source_domain=candidate.source_domain,
                candidate_image_url=image_url,
                face_similarity_score=best_score,
                is_verified_match=True,
                verification_status="VERIFIED_MATCH",
                best_face_bbox=best_bbox,
                raw_candidate=candidate
            )
        else:
            reason = f"Face similarity score ({best_score:.4f}) below verification threshold ({self.face_threshold:.4f})."
            logger.info(f"[REJECTED] {reason}")
            return CandidateVerificationResult(
                candidate_url=candidate.url,
                candidate_title=candidate.title,
                source_domain=candidate.source_domain,
                candidate_image_url=image_url,
                face_similarity_score=best_score,
                is_verified_match=False,
                verification_status="REJECTED_LOW_SIMILARITY",
                rejection_reason=reason,
                best_face_bbox=best_bbox,
                raw_candidate=candidate
            )

    def verify_search_candidates(
        self,
        query_image_or_embedding: Union[str, np.ndarray],
        candidates: List[SearchResultCandidate]
    ) -> List[CandidateVerificationResult]:
        """
        Verifies a list of search result candidates in ranked order.
        
        Args:
            query_image_or_embedding: Query image file path or raw 1D/2D embedding array.
            candidates: List of SearchResultCandidate instances.
            
        Returns:
            List of CandidateVerificationResult instances for tested candidates.
        """
        if isinstance(query_image_or_embedding, str):
            img, err = FaceIdentificationEngine.load_image(query_image_or_embedding)
            if img is None:
                logger.error(f"[VERIFICATION ERROR] Failed to load query image: {err}")
                return []
            faces = self.face_engine.extract_faces(img)
            if not faces:
                logger.error(f"[VERIFICATION ERROR] No faces detected in query image '{query_image_or_embedding}'")
                return []
            query_embedding = faces[0]["embedding"]
        else:
            query_embedding = query_image_or_embedding

        ranked_candidates = CandidateVerificationService.rank_candidates(candidates)
        results: List[CandidateVerificationResult] = []

        for candidate in ranked_candidates:
            res = self.verify_candidate(query_embedding, candidate)
            results.append(res)

            if res.is_verified_match and self.stop_on_first_match:
                logger.info("[PIPELINE COMPLETE] Verified match found. Stopping further candidate inspection.")
                break

        return results
