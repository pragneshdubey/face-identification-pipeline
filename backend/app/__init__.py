"""
Face Identification Pipeline App Package.
"""

from app.blockchain import (
    BaseBlockchainAdapter,
    Block,
    LocalBlockchainService,
    compute_sha256_fingerprint,
)
from app.face_engine import (
    FaceIdentificationEngine,
    RecognitionResult,
    compute_cosine_similarity,
    find_best_match,
    normalize_embedding,
)
from app.search_engine import (
    BaseReverseImageSearchProvider,
    CandidateFaceVerifier,
    CandidateVerificationResult,
    CandidateVerificationService,
    ConsentRegistrySearchProvider,
    MockReverseImageSearchProvider,
    SearchResponse,
    SearchResultCandidate,
    SerpApiGoogleLensProvider,
    extract_domain,
    is_social_media_domain,
    optimize_image_for_upload,
)

__all__ = [
    "FaceIdentificationEngine",
    "RecognitionResult",
    "compute_cosine_similarity",
    "find_best_match",
    "normalize_embedding",
    "BaseReverseImageSearchProvider",
    "ConsentRegistrySearchProvider",
    "SerpApiGoogleLensProvider",
    "MockReverseImageSearchProvider",
    "SearchResultCandidate",
    "SearchResponse",
    "CandidateVerificationService",
    "CandidateFaceVerifier",
    "CandidateVerificationResult",
    "extract_domain",
    "is_social_media_domain",
    "optimize_image_for_upload",
    "BaseBlockchainAdapter",
    "LocalBlockchainService",
    "Block",
    "compute_sha256_fingerprint",
]
