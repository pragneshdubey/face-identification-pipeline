"""
Face Identification Engine Module.

Uses InsightFace and ONNX Runtime for face detection, embedding extraction,
and cosine-similarity-based face identification.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class RecognitionResult:
    """Dataclass representing the result of a face recognition query."""
    person_id: Optional[str] = None
    similarity_score: float = 0.0
    is_match: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None  # (left, top, right, bottom)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    error_message: Optional[str] = None


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalizes a 1D or 2D feature embedding vector to unit L2 norm.
    
    Args:
        embedding: Input numpy array feature vector.
        
    Returns:
        L2 normalized numpy array.
    """
    norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
    if np.all(norm == 0):
        return embedding
    return embedding / norm


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two feature embedding vectors.
    
    Args:
        emb1: First embedding vector.
        emb2: Second embedding vector.
        
    Returns:
        Float cosine similarity score in range [-1.0, 1.0].
    """
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot_product = np.dot(emb1, emb2)
    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, -1.0, 1.0))


def find_best_match(
    query_emb: np.ndarray,
    known_embeddings: Dict[str, np.ndarray],
    threshold: float = 0.5
) -> RecognitionResult:
    """
    Compares a query face embedding against a database of known embeddings.
    
    Args:
        query_emb: Feature vector of the query face.
        known_embeddings: Dictionary mapping person_id to normalized embedding vector.
        threshold: Minimum similarity score required for a valid match.
        
    Returns:
        RecognitionResult object containing best matching identity and score.
    """
    if not known_embeddings:
        return RecognitionResult(
            person_id=None,
            similarity_score=0.0,
            is_match=False,
            embedding=query_emb,
            error_message="No known identities available in database."
        )

    best_person_id: Optional[str] = None
    best_score: float = -1.0

    for person_id, known_emb in known_embeddings.items():
        score = compute_cosine_similarity(query_emb, known_emb)
        if score > best_score:
            best_score = score
            best_person_id = person_id

    is_match = best_score >= threshold and best_person_id is not None

    return RecognitionResult(
        person_id=best_person_id if is_match else None,
        similarity_score=max(0.0, float(best_score)),
        is_match=is_match,
        embedding=query_emb,
        error_message=None if is_match else f"No match found above threshold {threshold:.2f} (best: {max(0.0, best_score):.2f})."
    )


class FaceIdentificationEngine:
    """
    Core Face Identification Engine wrapper using InsightFace.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        threshold: float = 0.5,
        providers: Optional[List[str]] = None,
        prepare_ctx_id: int = -1,
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initializes the InsightFace model for CPU execution.
        
        Args:
            model_name: Name of InsightFace model pack (default: 'buffalo_l').
            threshold: Cosine similarity threshold for identity matching.
            providers: Execution providers list (default: ['CPUExecutionProvider']).
            prepare_ctx_id: Context ID for InsightFace (-1 for CPU, >=0 for GPU).
            det_size: Detection resolution tuple (width, height).
        """
        self.model_name = model_name
        self.threshold = threshold
        self.providers = providers if providers is not None else ["CPUExecutionProvider"]
        self.prepare_ctx_id = prepare_ctx_id
        self.det_size = det_size

        self.known_embeddings: Dict[str, np.ndarray] = {}
        self._app = None  # Lazy loading InsightFace FaceAnalysis

    def _init_insightface(self):
        """Lazy initializer for InsightFace FaceAnalysis model."""
        if self._app is None:
            import insightface
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers
            )
            self._app.prepare(ctx_id=self.prepare_ctx_id, det_size=self.det_size)

    @staticmethod
    def load_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Loads an image from a file path using OpenCV.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (numpy image BGR array or None, error message string or None).
        """
        if not os.path.isfile(image_path):
            return None, f"Image file not found: '{image_path}'"

        img = cv2.imread(image_path)
        if img is None:
            return None, f"Failed to load image file (corrupt or invalid format): '{image_path}'"

        return img, None

    def extract_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detects faces in an image and extracts normalized embeddings.
        
        Args:
            image: BGR image numpy array.
            
        Returns:
            List of dictionaries containing 'bbox' and 'embedding'.
        """
        self._init_insightface()
        faces = self._app.get(image)
        results = []

        for face in faces:
            bbox = tuple(map(int, face.bbox.astype(int)))
            embedding = normalize_embedding(face.embedding)
            results.append({
                "bbox": bbox,
                "embedding": embedding,
                "det_score": float(face.det_score) if hasattr(face, 'det_score') else 1.0
            })

        return results

    def register_known_face(
        self,
        person_id: str,
        image_path: str
    ) -> Tuple[bool, str]:
        """
        Registers a known identity from an image file path into the in-memory database.
        
        Args:
            person_id: Unique string identifier for the individual.
            image_path: Path to the image file containing the individual's face.
            
        Returns:
            Tuple of (success boolean, status message).
        """
        img, err = self.load_image(image_path)
        if img is None:
            return False, err or "Image loading error."

        faces = self.extract_faces(img)
        if not faces:
            return False, f"No face detected in image for person_id '{person_id}'."

        if len(faces) > 1:
            # If multiple faces detected, pick the face with largest bounding box area
            faces.sort(
                key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
                reverse=True
            )

        self.known_embeddings[person_id] = faces[0]["embedding"]
        return True, f"Successfully registered face for person_id '{person_id}'."

    def load_known_faces_directory(self, directory_path: str) -> Dict[str, bool]:
        """
        Scans a directory for images and registers each as a known face identity.
        Supports file formats: .jpg, .jpeg, .png, .bmp, .webp.
        The filename (without extension) is used as the person_id.
        
        Args:
            directory_path: Directory path containing target face images.
            
        Returns:
            Dictionary mapping person_id to registration success boolean.
        """
        registration_results: Dict[str, bool] = {}
        if not os.path.isdir(directory_path):
            return registration_results

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for filename in os.listdir(directory_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_exts:
                person_id = os.path.splitext(filename)[0]
                image_path = os.path.join(directory_path, filename)
                success, _ = self.register_known_face(person_id, image_path)
                registration_results[person_id] = success

        return registration_results

    def identify_faces(
        self,
        image_path: str,
        threshold: Optional[float] = None
    ) -> List[RecognitionResult]:
        """
        Identifies all faces present in a query image.
        
        Args:
            image_path: Path to the test image file.
            threshold: Optional custom threshold to override default threshold.
            
        Returns:
            List of RecognitionResult instances (one per detected face).
        """
        match_threshold = threshold if threshold is not None else self.threshold

        img, err = self.load_image(image_path)
        if img is None:
            return [RecognitionResult(
                person_id=None,
                similarity_score=0.0,
                is_match=False,
                error_message=err
            )]

        faces = self.extract_faces(img)
        if not faces:
            return [RecognitionResult(
                person_id=None,
                similarity_score=0.0,
                is_match=False,
                error_message=f"No faces detected in image '{image_path}'."
            )]

        results = []
        for face in faces:
            result = find_best_match(
                query_emb=face["embedding"],
                known_embeddings=self.known_embeddings,
                threshold=match_threshold
            )
            result.bbox = face["bbox"]
            results.append(result)

        return results

