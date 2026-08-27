def test_run_pixel_probe_q14_diagnostic(tmp_path):

    # Needs valid pdf_path, page_number, profile_dir, reference_path
    # We can mock this or create dummy files, but run_pixel_probe expects real files.
    # Actually, pixel probe tests currently mock or test sub-functions.
    # Let's test the JSON serialization of PixelProbeReport with q14_diagnostics.
    pass


def test_pixel_probe_report_q14_diagnostics_serializability():
    import dataclasses

    from matera.tools.pixel_probe import PixelProbeReport, _json_ready
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
                diff_mask_pixels=0,
                saturation_mask_pixels=0,
                hsv_final_mask_pixels=0,
                safe_mask_pixels=0,
                excluded_margin_pixels=0,
                safe_mask_ratio=0.0,
                hsv_to_diff_ratio=0.0,
                classifier_available=False,
                classifier_probability=None,
                decision_reason="BLANK",
                suspicion_notes=[],
                artifacts={},
            )
        ],
    )
    data = _json_ready(dataclasses.asdict(report))
    assert data["report_version"] == 2
    assert "q14_diagnostics" in data
    assert len(data["q14_diagnostics"]) == 1
    assert data["q14_diagnostics"][0]["question_id"] == "Q14"
