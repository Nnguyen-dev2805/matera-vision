import cv2
import numpy as np
from PIL import Image

from matera.data.contracts import RenderedPage
from matera.vision.contracts import AlignedPage, AlignmentConfig, AlignmentError


def align_page(
    source_page: RenderedPage,
    reference_image: Image.Image,
    config: AlignmentConfig,
    profile_form_id: str = "unknown",
    profile_version: str = "unknown",
) -> AlignedPage:
    """
    Aligns the source page to the reference image using feature matching.
    The warp_matrix transforms from source space to reference space.
    Raises AlignmentError if the alignment_score (inlier ratio) < threshold.
    """
    if config.algorithm != "orb":
        raise NotImplementedError(f"Algorithm '{config.algorithm}' is not supported.")

    source_arr = np.array(source_page.image)
    ref_arr = np.array(reference_image)

    if source_arr.size == 0 or ref_arr.size == 0:
        raise AlignmentError(
            "Input images cannot be empty.",
            page_number=source_page.page_number,
            alignment_score=0.0,
        )

    # Convert to grayscale
    if len(source_arr.shape) == 3:
        source_gray = cv2.cvtColor(source_arr, cv2.COLOR_RGB2GRAY)
    else:
        source_gray = source_arr

    if len(ref_arr.shape) == 3:
        ref_gray = cv2.cvtColor(ref_arr, cv2.COLOR_RGB2GRAY)
    else:
        ref_gray = ref_arr

    orb = cv2.ORB_create(nfeatures=config.max_features)

    kp1, des1 = orb.detectAndCompute(source_gray, None)
    kp2, des2 = orb.detectAndCompute(ref_gray, None)

    if not kp1 or not kp2 or des1 is None or des2 is None:
        raise AlignmentError(
            "Could not extract sufficient features for alignment.",
            page_number=source_page.page_number,
            alignment_score=0.0,
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if not matches:
        raise AlignmentError(
            "No feature matches found between source and reference.",
            page_number=source_page.page_number,
            alignment_score=0.0,
        )

    matches = sorted(matches, key=lambda x: x.distance)
    # Take top matches to improve RANSAC speed and exclude extreme outliers
    good_matches = matches[: min(len(matches), 1000)]

    if len(good_matches) < 4:
        raise AlignmentError(
            f"Not enough good matches ({len(good_matches)}) found for transformation.",
            page_number=source_page.page_number,
            alignment_score=0.0,
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Set RNG seed for deterministic RANSAC results
    cv2.setRNGSeed(42)
    
    warp_matrix = None
    inliers = None
    if config.transform_model == "affine":
        warp_matrix, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC)
    elif config.transform_model == "similarity":
        warp_matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    else:
        raise NotImplementedError(f"Transform model '{config.transform_model}' is not supported.")

    if warp_matrix is None or inliers is None:
        raise AlignmentError(
            "Transformation matrix computation failed.",
            page_number=source_page.page_number,
            alignment_score=0.0,
        )

    inlier_count = int(np.sum(inliers))
    total_matches = len(good_matches)
    alignment_score = inlier_count / total_matches if total_matches > 0 else 0.0

    if alignment_score < config.inlier_threshold:
        raise AlignmentError(
            f"Alignment score {alignment_score:.3f} is below threshold {config.inlier_threshold}.",
            page_number=source_page.page_number,
            alignment_score=alignment_score,
        )

    target_h, target_w = ref_gray.shape
    if config.target_width > 0 and config.target_height > 0:
        target_w = config.target_width
        target_h = config.target_height

    aligned_arr = cv2.warpAffine(
        source_arr,
        warp_matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    if len(source_arr.shape) == 3:
        aligned_image = Image.fromarray(aligned_arr, mode="RGB")
    else:
        aligned_image = Image.fromarray(aligned_arr, mode="L")

    return AlignedPage(
        page_number=source_page.page_number,
        image=aligned_image,
        profile_form_id=profile_form_id,
        profile_version=profile_version,
        reference_dpi=config.reference_dpi,
        warp_matrix=warp_matrix,
        alignment_score=alignment_score,
    )
