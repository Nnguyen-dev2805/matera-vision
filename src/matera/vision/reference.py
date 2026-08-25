from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from matera.vision.contracts import AlignedPage

logger = logging.getLogger(__name__)

def generate_median_reference(aligned_pages: list["AlignedPage"]) -> Image.Image:
    """
    Generates a clean "blank" reference page by calculating the median pixel value
    across a batch of aligned pages.
    
    Because handwriting (ink) varies in position across different pages, the median
    filter will effectively erase the ink, leaving only the stationary printed text
    and the background paper color.
    
    Args:
        aligned_pages: List of AlignedPage objects, all aligned to the same coordinate space.
        
    Returns:
        A synthesized blank Image.Image.
    """
    if not aligned_pages:
        raise ValueError("Cannot generate median reference from an empty list of pages.")
        
    logger.info(f"Generating median reference from {len(aligned_pages)} aligned pages...")
    
    # Check that all images have the same dimensions
    first_size = aligned_pages[0].image.size
    for page in aligned_pages:
        if page.image.size != first_size:
            logger.warning(f"Image size mismatch in median generation: {page.image.size} != {first_size}. Resizing...")
            page.image = page.image.resize(first_size, Image.Resampling.LANCZOS)
            
    # Convert all PIL images to numpy arrays
    # Stacking 10 large images requires ~250MB RAM, which is acceptable.
    arrays = []
    for page in aligned_pages:
        # Convert to BGR format for consistency with OpenCV processing later
        img_np = np.array(page.image)
        # Handle grayscale or RGBA if present
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        arrays.append(img_np)
        
    # Stack along a new axis (0) and calculate median
    # Use np.median which is memory intensive but exact.
    stacked = np.stack(arrays, axis=0)
    median_np = np.median(stacked, axis=0).astype(np.uint8)
    
    logger.info("Median reference generation complete.")
    
    # Convert back to PIL Image (RGB)
    return Image.fromarray(median_np)
