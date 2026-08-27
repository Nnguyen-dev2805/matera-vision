import numpy as np
from PIL import Image

from matera.core.layout import BoundingBox, RoiDef
from matera.tools.q14_diagnostic import Q14CheckboxDiagnostic, compute_q14_checkbox_diagnostic


def test_compute_q14_checkbox_diagnostic():
    # Setup dummy data
    aligned_img = Image.new("RGB", (100, 100), color="white")
    reference_img = Image.new("RGB", (100, 100), color="white")
    median_ref_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 255
    roi = RoiDef(question_id="Q14", option_id="a", bbox=BoundingBox(x=10, y=10, w=20, h=20))

    # Test blank (no diff)
    diag = compute_q14_checkbox_diagnostic(
        aligned_image=aligned_img,
        reference_image=reference_img,
        median_ref_bgr=median_ref_bgr,
        roi=roi,
        trace=None,
        expected_state="BLANK",
        output_dir=None,
    )

    assert isinstance(diag, Q14CheckboxDiagnostic)
    assert diag.actual_state == "BLANK"
    assert diag.safe_mask_pixels < 20
    assert "BLANK" in diag.decision_reason
