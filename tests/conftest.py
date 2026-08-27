import sys
import types

import pytest


class DummyClf:
    def predict_proba(self, X):
        return [0.9]


@pytest.fixture
def mock_classifier(monkeypatch):
    """Provides a dummy classifier and mocks matera.classifier.model to prevent sys.modules pollution."""
    mod = types.ModuleType("matera.classifier.model")
    mod.extract_hog_features = lambda x: [0.1]

    class DummyAmbiguityClassifier:
        @staticmethod
        def load(path):
            return DummyClf()

    mod.AmbiguityClassifier = DummyAmbiguityClassifier
    monkeypatch.setitem(sys.modules, "matera.classifier.model", mod)

    return DummyClf()


@pytest.fixture
def mock_text_bbox(monkeypatch):
    """Mocks matera.vision.mark.get_text_bounding_box for tests that need it."""
    import matera.vision.mark as mark

    monkeypatch.setattr(mark, "get_text_bounding_box", lambda x: (5, 5, 30, 30))
