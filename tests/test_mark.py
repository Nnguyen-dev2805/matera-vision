import numpy as np
import pytest
from PIL import Image

from matera.vision.mark import calculate_features, create_mark_map, normalize_score


def _create_synthetic_patch(mode: str) -> Image.Image:
    """Helper to create synthetic RGB patches for testing."""
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)  # White background

    # Draw a black printed box representing the form
    arr[10:90, 10:15, :] = 0
    arr[10:90, 85:90, :] = 0
    arr[10:15, 10:90, :] = 0
    arr[85:90, 10:90, :] = 0

    if mode == "checked":
        # Draw a dark blue handwritten checkmark
        arr[40:60, 40:60, :] = [0, 0, 128]
    elif mode == "noisy_aligned":
        # Simulate slight alignment noise (shift box by 1 pixel)
        arr = np.full((100, 100, 3), 255, dtype=np.uint8)
        arr[11:91, 11:16, :] = 0
        arr[11:91, 86:91, :] = 0
        arr[11:16, 11:91, :] = 0
        arr[86:91, 11:91, :] = 0

    return Image.fromarray(arr)


def test_create_mark_map_suppression():
    """When source and reference are identical, the map should be completely empty."""
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("clean")

    mask = create_mark_map(src, ref)
    mask_arr = np.array(mask)

    assert mask_arr.shape == (100, 100)
    assert np.count_nonzero(mask_arr) == 0


def test_create_mark_map_checked():
    """A checked box should reveal only the checkmark, suppressing the printed box."""
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("checked")

    mask = create_mark_map(src, ref)
    mask_arr = np.array(mask)

    # The checkmark is 20x20 = 400 pixels
    # Since morphology might eat a tiny bit of the boundary or expand it slightly,
    # we expect the area to be roughly 400.
    foreground_pixels = np.count_nonzero(mask_arr)
    assert 300 < foreground_pixels < 500

    # Ensure the printed box (which is near the edges) is suppressed
    # Edges 0-20 and 80-100 should be completely clean
    assert np.count_nonzero(mask_arr[0:20, :]) == 0
    assert np.count_nonzero(mask_arr[80:100, :]) == 0


def test_create_mark_map_minor_noise_suppression():
    """Minor misalignment (1px shift) should be mostly suppressed by morphology."""
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("noisy_aligned")

    mask = create_mark_map(src, ref)
    mask_arr = np.array(mask)

    # A 1-pixel shift of a 80x80 box boundary creates a lot of difference,
    # but morphology open should kill most 1-pixel thick noise.
    # Total pixels is 10000. 0.005 ratio is 50 pixels.
    # We expect near-zero noise.
    assert np.count_nonzero(mask_arr) < 50


def test_create_mark_map_dimension_mismatch():
    ref = Image.new("RGB", (100, 100), "white")
    src = Image.new("RGB", (100, 110), "white")

    with pytest.raises(ValueError, match="same dimensions"):
        create_mark_map(src, ref)


def test_create_mark_map_immutability():
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("checked")

    ref_arr_before = np.array(ref)
    src_arr_before = np.array(src)

    create_mark_map(src, ref)

    np.testing.assert_array_equal(np.array(ref), ref_arr_before)
    np.testing.assert_array_equal(np.array(src), src_arr_before)


def test_calculate_features_blank():
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("clean")
    mask = create_mark_map(src, ref)

    features = calculate_features(src, mask)

    # 10000 total pixels. The box is printed in black so dark pixels might be some small amount,
    # but let's just check the foreground area ratio which should be 0.
    assert features.foreground_area_ratio == 0.0
    assert features.contour_count == 0
    assert features.largest_component_ratio == 0.0
    assert features.bbox_fill_ratio == 0.0

    score = normalize_score(features, "circle")
    assert score == 0.0


def test_calculate_features_checked():
    ref = _create_synthetic_patch("clean")
    src = _create_synthetic_patch("checked")
    mask = create_mark_map(src, ref)

    features = calculate_features(src, mask)

    # Checkmark is 20x20 = 400 pixels out of 10000 = 0.04
    assert 0.03 < features.foreground_area_ratio < 0.05
    assert features.contour_count >= 1
    assert features.largest_component_ratio > 0.02
    assert features.bbox_fill_ratio > 0.5  # It's a solid block in the synthetic test

    score = normalize_score(features, "circle")
    # Score should be ~ 0.04 * 10 = 0.4
    assert 0.3 < score < 0.5
