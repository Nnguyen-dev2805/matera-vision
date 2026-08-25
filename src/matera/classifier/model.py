import pickle
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from a binary mask image.
    Used by the V17 Ambiguity Classifier (SVM).
    """
    img_resized = cv2.resize(img, (24, 24))
    features = hog(
        img_resized, 
        orientations=8, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=False, 
        feature_vector=True
    )
    return features

class AmbiguityClassifier:
    """
    A wrapper around sklearn SVC for classifying ambiguous ROIs using HOG features (V17 pipeline).
    """
    def __init__(self) -> None:
        self.model = None
        self.is_trained = False

    def predict_proba(self, X: list[np.ndarray]) -> list[float]:
        """
        X: List of feature vectors (e.g., from extract_hog_features)
        Returns: List of probabilities (float between 0 and 1) representing the "MARKED" class.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be loaded/trained before predicting.")
        
        probs = self.model.predict_proba(X)
        # Assuming class 1 is MARKED
        return [float(p[1]) for p in probs]

    def predict(self, X: list[np.ndarray]) -> list[int]:
        """
        X: List of feature vectors.
        Returns: List of predicted classes (0 or 1).
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be loaded/trained before predicting.")
        return [int(p) for p in self.model.predict(X)]

    @classmethod
    def load(cls, path: Path | str) -> "AmbiguityClassifier":
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"Model file not found at {in_path}")
            
        instance = cls()
        with open(in_path, "rb") as f:
            instance.model = pickle.load(f)
        instance.is_trained = True
        return instance

