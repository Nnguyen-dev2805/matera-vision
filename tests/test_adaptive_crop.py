from matera.vision.adaptive_crop import GlobalCropConfig, EdgeInkStats, GlobalCropIteration, GlobalCropResult

def test_config_defaults():
    config = GlobalCropConfig()
    assert config.base_pad == 20
    assert config.expansion_step_px == 20
    assert config.max_pad_px == 60
    assert config.edge_band_px == 4
    assert config.edge_ink_threshold_px == 20
    assert config.edge_contour_min_area == 30.0
    assert config.max_iterations == 5

import pytest
import numpy as np
from PIL import Image
from matera.core.layout import RoiDef, BoundingBox
from matera.vision.adaptive_crop import compute_adaptive_global_crop, GlobalCropConfig

def test_adaptive_crop_base_case():
    config = GlobalCropConfig(base_pad=10, max_pad_px=30, expansion_step_px=10)
    aligned_img = Image.new("RGB", (200, 200), color="white")
    median_ref = np.full((200, 200, 3), 255, dtype=np.uint8)
    
    rois = [
        RoiDef("Q1", "A", BoundingBox(50, 50, 20, 20)),
        RoiDef("Q1", "B", BoundingBox(50, 80, 20, 20)),
    ]
    
    result = compute_adaptive_global_crop(aligned_img, median_ref, rois, config)
    assert not result.expanded
    assert result.stop_reason == "no_edge_pressure"
    assert result.pads["left"] == 10
    assert result.pads["right"] == 10
    assert result.pads["top"] == 10
    assert result.pads["bottom"] == 10
    assert result.crop["x1"] == 40
    assert result.crop["y1"] == 40
    assert result.crop["x2"] == 80
    assert result.crop["y2"] == 110

def test_adaptive_crop_left_edge_pressure():
    config = GlobalCropConfig(base_pad=10, max_pad_px=30, expansion_step_px=10)
    
    aligned_img = Image.new("RGB", (200, 200), color="white")
    arr = np.array(aligned_img)
    # Draw dark ink on the left side (x=40..45, y=60..80) that touches the base crop edge (x=40)
    arr[60:80, 40:45] = [0, 0, 0]
    aligned_img = Image.fromarray(arr)
    
    median_ref = np.full((200, 200, 3), 255, dtype=np.uint8)
    
    rois = [
        RoiDef("Q1", "A", BoundingBox(50, 50, 20, 20)),
        RoiDef("Q1", "B", BoundingBox(50, 80, 20, 20)),
    ]
    
    result = compute_adaptive_global_crop(aligned_img, median_ref, rois, config)
    assert result.expanded
    assert result.pads["left"] > 10
    assert result.pads["right"] == 10
    assert result.crop["x1"] < 40

def test_adaptive_crop_max_pad():
    config = GlobalCropConfig(base_pad=10, max_pad_px=30, expansion_step_px=10)
    
    aligned_img = Image.new("RGB", (200, 200), color="white")
    arr = np.array(aligned_img)
    # Draw dark ink reaching all the way to x=10 (beyond max_pad_px which allows x1 down to 20)
    arr[60:80, 10:45] = [0, 0, 0]
    aligned_img = Image.fromarray(arr)
    
    median_ref = np.full((200, 200, 3), 255, dtype=np.uint8)
    
    rois = [
        RoiDef("Q1", "A", BoundingBox(50, 50, 20, 20)),
    ]
    
    result = compute_adaptive_global_crop(aligned_img, median_ref, rois, config)
    assert result.expanded
    assert result.pads["left"] == 30
    assert result.stop_reason in ("max_pad_reached", "no_edge_pressure")
