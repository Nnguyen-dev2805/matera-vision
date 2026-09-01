import cv2
import numpy as np
import pytest
from PIL import Image

from matera.data.contracts import RenderedPage
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, AlignmentError


@pytest.fixture
def base_config():
    return AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.5,
        reference_dpi=300,
    )


@pytest.fixture
def reference_image():
    # Create a dummy image with random noise so ORB can find keypoints
    np.random.seed(42)  # For determinism
    img_arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    # Add a white square in the middle to give some strong corners
    img_arr[100:200, 100:200] = 255
    return Image.fromarray(img_arr, mode="RGB")


@pytest.fixture
def source_page(reference_image):
    return RenderedPage(
        page_number=1,
        width_px=300,
        height_px=300,
        pdf_width_pt=200.0,
        pdf_height_pt=200.0,
        image=reference_image.copy(),
    )


def test_alignment_perfect_match(source_page, reference_image, base_config):
    result = align_page(source_page, reference_image, base_config, "test_form", "v1")

    assert result.page_number == 1
    assert result.alignment_score >= 0.99

    expected_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(result.warp_matrix, expected_matrix, atol=1e-3)


def test_alignment_translation(source_page, reference_image, base_config):
    # Translate the source image by dx=20, dy=-15
    dx, dy = 20, -15
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    src_arr = np.array(source_page.image)
    shifted_arr = cv2.warpAffine(src_arr, M, (300, 300), borderValue=(255, 255, 255))

    # Update the source page with the shifted image
    shifted_page = RenderedPage(
        page_number=source_page.page_number,
        width_px=source_page.width_px,
        height_px=source_page.height_px,
        pdf_width_pt=source_page.pdf_width_pt,
        pdf_height_pt=source_page.pdf_height_pt,
        image=Image.fromarray(shifted_arr),
    )

    result = align_page(shifted_page, reference_image, base_config, "test_form", "v1")

    assert result.alignment_score > 0.6
    # The calculated warp matrix should reverse the shift, i.e., -dx, -dy
    # Because it maps from source (shifted) to reference (original)
    expected_matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
    np.testing.assert_allclose(result.warp_matrix, expected_matrix, atol=1.0)


def test_alignment_rotation(source_page, reference_image, base_config):
    # Rotate the source image by 2 degrees
    angle = 2.0
    center = (150, 150)
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    src_arr = np.array(source_page.image)
    rotated_arr = cv2.warpAffine(src_arr, M, (300, 300), borderValue=(255, 255, 255))

    rotated_page = RenderedPage(
        page_number=source_page.page_number,
        width_px=source_page.width_px,
        height_px=source_page.height_px,
        pdf_width_pt=source_page.pdf_width_pt,
        pdf_height_pt=source_page.pdf_height_pt,
        image=Image.fromarray(rotated_arr),
    )

    result = align_page(rotated_page, reference_image, base_config, "test_form", "v1")

    assert result.alignment_score > 0.6

    # The recovered matrix should be the inverse rotation (angle = -2.0)
    expected_M_inv = cv2.getRotationMatrix2D(center, -angle, scale=1.0)
    np.testing.assert_allclose(result.warp_matrix, expected_M_inv, atol=1.0)


def test_alignment_scale(source_page, reference_image, base_config):
    # Scale the source image by 1.05
    scale = 1.05
    center = (150, 150)
    M = cv2.getRotationMatrix2D(center, 0, scale=scale)

    src_arr = np.array(source_page.image)
    scaled_arr = cv2.warpAffine(src_arr, M, (300, 300), borderValue=(255, 255, 255))

    scaled_page = RenderedPage(
        page_number=source_page.page_number,
        width_px=source_page.width_px,
        height_px=source_page.height_px,
        pdf_width_pt=source_page.pdf_width_pt,
        pdf_height_pt=source_page.pdf_height_pt,
        image=Image.fromarray(scaled_arr),
    )

    result = align_page(scaled_page, reference_image, base_config, "test_form", "v1")

    assert result.alignment_score > 0.6

    # The recovered matrix should be the inverse scale (1/1.05)
    expected_M_inv = cv2.getRotationMatrix2D(center, 0, scale=1.0 / scale)
    np.testing.assert_allclose(result.warp_matrix, expected_M_inv, atol=1.0)


def test_alignment_blank_page_fails(source_page, reference_image, base_config):
    # Create a solid white image (no features)
    blank_arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    blank_page = RenderedPage(
        page_number=source_page.page_number,
        width_px=source_page.width_px,
        height_px=source_page.height_px,
        pdf_width_pt=source_page.pdf_width_pt,
        pdf_height_pt=source_page.pdf_height_pt,
        image=Image.fromarray(blank_arr),
    )

    with pytest.raises(AlignmentError) as exc_info:
        align_page(blank_page, reference_image, base_config, "test_form", "v1")

    assert "Could not extract sufficient features" in str(exc_info.value)
    assert exc_info.value.alignment_score == 0.0


def test_alignment_immutability(source_page, reference_image, base_config):
    original_arr = np.array(source_page.image).copy()

    # Perform alignment
    align_page(source_page, reference_image, base_config, "test_form", "v1")

    # Verify the source page image is completely unmodified
    current_arr = np.array(source_page.image)
    np.testing.assert_array_equal(current_arr, original_arr)


def test_alignment_unsupported_algorithm(source_page, reference_image):
    bad_config = AlignmentConfig(algorithm="sift")  # type: ignore
    with pytest.raises(NotImplementedError):
        align_page(source_page, reference_image, bad_config, "test_form", "v1")


def test_alignment_unsupported_transform(source_page, reference_image):
    bad_config = AlignmentConfig(transform_model="homography")  # type: ignore
    with pytest.raises(NotImplementedError):
        align_page(source_page, reference_image, bad_config, "test_form", "v1")


def test_alignment_below_threshold(source_page, reference_image):
    # Set threshold unrealistically high
    strict_config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.999,
        reference_dpi=300,
    )

    # Create an image with the same features but scrambled geometry
    # Swap the left and right halves of the image.
    # Descriptors will match, but RANSAC can only fit at most 50% of them.
    src_arr = np.array(source_page.image)
    h, w, c = src_arr.shape
    scrambled_arr = np.empty_like(src_arr)
    half_w = w // 2
    scrambled_arr[:, :half_w] = src_arr[:, half_w:]
    scrambled_arr[:, half_w:] = src_arr[:, :half_w]

    scrambled_page = RenderedPage(
        page_number=source_page.page_number,
        width_px=source_page.width_px,
        height_px=source_page.height_px,
        pdf_width_pt=source_page.pdf_width_pt,
        pdf_height_pt=source_page.pdf_height_pt,
        image=Image.fromarray(scrambled_arr),
    )

    with pytest.raises(AlignmentError) as exc_info:
        align_page(scrambled_page, reference_image, strict_config, "test_form", "v1")

    assert "is below threshold" in str(exc_info.value)
    assert exc_info.value.alignment_score < 0.999
    assert exc_info.value.alignment_score > 0.0


def test_alignment_determinism(source_page, reference_image, base_config):
    # Run alignment 3 times and ensure results are byte-for-byte identical
    result1 = align_page(source_page, reference_image, base_config)
    result2 = align_page(source_page, reference_image, base_config)
    result3 = align_page(source_page, reference_image, base_config)

    np.testing.assert_array_equal(result1.warp_matrix, result2.warp_matrix)
    np.testing.assert_array_equal(result2.warp_matrix, result3.warp_matrix)

    img1_arr = np.array(result1.image)
    img2_arr = np.array(result2.image)
    img3_arr = np.array(result3.image)
    np.testing.assert_array_equal(img1_arr, img2_arr)
    np.testing.assert_array_equal(img2_arr, img3_arr)


def test_alignment_dimension_mismatch(source_page, base_config):
    # Create a reference image that is radically different in size/aspect ratio
    # but has no common features
    bad_ref_arr = np.full((10, 1000, 3), 128, dtype=np.uint8)
    bad_ref_image = Image.fromarray(bad_ref_arr)

    with pytest.raises(AlignmentError):
        align_page(source_page, bad_ref_image, base_config)


def test_alignment_target_size(source_page, reference_image):
    # Test that the output is cropped/padded exactly to target_width/height
    config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.5,
        target_width=400,
        target_height=500,
    )
    result = align_page(source_page, reference_image, config)
    assert result.image.width == 400
    assert result.image.height == 500
