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
