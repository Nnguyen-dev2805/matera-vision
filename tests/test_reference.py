import numpy as np
import pytest
from PIL import Image

from matera.vision.contracts import AlignedPage
from matera.vision.reference import generate_median_reference


def test_generate_median_reference():
    img1_cv = np.zeros((100, 100, 3), dtype=np.uint8)
    img1_cv.fill(255)
    img1_cv[10:20, 10:20] = [0, 0, 0] # Ink on img1
    
    img2_cv = np.zeros((100, 100, 3), dtype=np.uint8)
    img2_cv.fill(255)
    img2_cv[30:40, 30:40] = [0, 0, 0] # Ink on img2
    
    img3_cv = np.zeros((100, 100, 3), dtype=np.uint8)
    img3_cv.fill(255)
    img3_cv[50:60, 50:60] = [0, 0, 0] # Ink on img3
    
    pages = [
        AlignedPage(page_number=1, image=Image.fromarray(img1_cv), profile_form_id="test", profile_version="v1", reference_dpi=300, warp_matrix=np.eye(3), alignment_score=1.0),
        AlignedPage(page_number=2, image=Image.fromarray(img2_cv), profile_form_id="test", profile_version="v1", reference_dpi=300, warp_matrix=np.eye(3), alignment_score=1.0),
        AlignedPage(page_number=3, image=Image.fromarray(img3_cv), profile_form_id="test", profile_version="v1", reference_dpi=300, warp_matrix=np.eye(3), alignment_score=1.0)
    ]
    
    median_img = generate_median_reference(pages)
    assert median_img.size == (100, 100)
    
    median_cv = np.array(median_img)
    # The ink should be removed because it's only in 1 out of 3 images (median of 255, 255, 0 is 255)
    assert np.all(median_cv[10:20, 10:20] == 255)
    assert np.all(median_cv[30:40, 30:40] == 255)
    assert np.all(median_cv[50:60, 50:60] == 255)

def test_generate_median_reference_empty():
    with pytest.raises(ValueError):
        generate_median_reference([])
