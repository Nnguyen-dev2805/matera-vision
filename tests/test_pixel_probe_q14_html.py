def test_q14_html_rendering(tmp_path):
    from matera.tools.pixel_probe import PixelProbeReport, _write_index_html
    from matera.tools.q14_diagnostic import Q14CheckboxDiagnostic

    report = PixelProbeReport(
        report_version=2,
        source_pdf="test.pdf",
        page_number=1,
        alignment_score=1.0,
        warp_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        filters={"question": None, "option": None, "routing_evaluated": True},
        thresholds={},
        questions=[],
        global_topology={},
        traces=[],
        artifacts={},
        q14_diagnostics=[
            Q14CheckboxDiagnostic(
                page_number=1,
                question_id="Q14",
                option_id="a",
                expected_state="BLANK",
                actual_state="BLANK",
                score=1.0,
                method="HSV_AI_L2",
                routing_status=None,
                bbox={"x": 10, "y": 10, "w": 20, "h": 20},
                crop_coords={"x1": 10, "y1": 10, "x2": 30, "y2": 30},
                diff_mask_pixels=10,
                saturation_mask_pixels=5,
                hsv_final_mask_pixels=4,
                safe_mask_pixels=2,
                excluded_margin_pixels=2,
                safe_mask_ratio=0.5,
                hsv_to_diff_ratio=0.4,
                classifier_available=False,
                classifier_probability=None,
                decision_reason="BLANK",
                suspicion_notes=["test note"],
                artifacts={"q14_crop_aligned.png": "path.png"},
            )
        ],
        evidence=None,
    )

    _write_index_html(tmp_path, report)
    html_content = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Q14 Checkbox HSV Diagnostic" in html_content
    assert "safe_mask_pixels" in html_content
