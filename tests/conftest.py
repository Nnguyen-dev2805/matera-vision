import pytest


@pytest.fixture
def mock_text_bbox(monkeypatch):
    """Mocks matera.vision.detectors.local_geometry.get_text_bounding_box for tests that need it."""
    import matera.vision.detectors.local_geometry as mark_radial

    monkeypatch.setattr(mark_radial, "get_text_bounding_box", lambda x: (5, 5, 30, 30))
