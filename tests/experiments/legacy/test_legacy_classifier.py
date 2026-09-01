import numpy as np

from matera.models.legacy_classifier.classifier import AmbiguityClassifier
from matera.models.legacy_classifier.hog import extract_hog_features


def test_extract_hog_features():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 255
    features = extract_hog_features(img)
    assert features.shape == (128,)


def test_ambiguity_classifier_uninitialized():
    clf = AmbiguityClassifier()
    assert not clf.is_trained
    import pytest

    with pytest.raises(RuntimeError):
        clf.predict([np.zeros(128)])

    with pytest.raises(RuntimeError):
        clf.predict_proba([np.zeros(128)])
