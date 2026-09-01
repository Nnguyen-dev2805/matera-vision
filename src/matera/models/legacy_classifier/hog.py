import cv2
import numpy as np
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
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return features
