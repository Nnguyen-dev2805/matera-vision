import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.contracts import RenderedPage
from matera.data.extract import extract_pages
from matera.evaluation.ground_truth import expand_ground_truth_labels, load_ground_truth
from matera.evaluation.ground_truth_metrics import evaluate_page
from matera.evaluation.ground_truth_report import GroundTruthReporter
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.evidence import evidence_to_mark_scores, extract_mark_evidence
from matera.vision.routing import route_page


def get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def get_git_dirty():
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1]
            )
            .decode("utf-8")
            .strip()
        )
        return len(status) > 0
    except Exception:
        return False


def run_pixel_probe_cli(
    pdf_path: Path,
    page_num: int,
    output_dir: Path,
    profile_path: Path,
    reference_path: Path,
    question: str = None,
    alignment_config_dict: dict = None,
    routing_config_dict: dict = None,
):
    """Invokes the pixel probe CLI to generate debug artifacts."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "debug_pixel_pipeline.py"),
        "--pdf",
        str(pdf_path),
        "--page",
        str(page_num),
        "--output-dir",
        str(output_dir),
        "--profile-dir",
        str(profile_path.parent),
        "--reference",
        str(reference_path),
        "--dpi",
        "300",
    ]
    if question:
        cmd.extend(["--question", question])
    if alignment_config_dict:
        cmd.extend(["--alignment-json", json.dumps(alignment_config_dict)])
    if routing_config_dict:
        cmd.extend(["--routing-json", json.dumps(routing_config_dict)])

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        htmls = list(output_dir.rglob("index.html"))
        return htmls[0] if htmls else None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        print(f"Error running pixel_probe: {stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate ground truth labels")
    parser.add_argument(
        "--ground-truth", type=Path, required=True, help="Path to ground truth JSON"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--debug-artifacts", action="store_true", help="Generate debug artifacts")
    parser.add_argument("--max-items", type=int, help="Max items to process")

    parser.add_argument("--question", type=str, help="Optional question filter, e.g. Q4")
    parser.add_argument("--page", type=int, help="Optional page number filter")
    parser.add_argument("--pdf-substring", type=str, help="Optional PDF name filter")
    parser.add_argument(
        "--reference-type",
        type=str,
        choices=["golden", "hybrid"],
        default="golden",
        help="Type of reference image to use",
    )
    parser.add_argument("--profile", type=Path, help="Override semantic profile path")
    parser.add_argument("--layout", type=Path, help="Override layout profile path")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")

    args = parser.parse_args()

    gt_path = args.ground_truth.resolve()
    base_out_dir = args.output_dir.resolve()

    if not gt_path.exists():
        print(f"Ground truth file not found: {gt_path}")
        sys.exit(1)

    dataset = load_ground_truth(gt_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = timestamp
    run_dir = base_out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "run_id": run_id,
        "ground_truth_file": str(gt_path),
        "output_dir": str(run_dir),
        "debug_artifacts": args.debug_artifacts,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "timestamp": timestamp,
        "alignment": {"algorithm": "orb", "transform_model": "affine", "inlier_threshold": 0.05},
        "routing": {"low_threshold": 0.2, "high_threshold": 0.6},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    # Write run_config to JSON
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    alignment_config = AlignmentConfig(**run_config["alignment"])
    routing_config = RoutingConfig(**run_config["routing"])

    page_results = []

    items_to_process = dataset.items

    if args.pdf_substring:
        items_to_process = [i for i in items_to_process if args.pdf_substring in i.pdf_path.name]
    if args.page:
        items_to_process = [i for i in items_to_process if i.page_num == args.page]

    if args.max_items:
        items_to_process = items_to_process[: args.max_items]

    # Global golden reference
    golden_ref_path = (
        Path(__file__).resolve().parents[1] / "scratch" / "synthetic_median_reference.png"
    )

    for item in items_to_process:
        print(f"Processing {item.item_id} (PDF: {item.pdf_path.name}, Page: {item.page_num})")

        # Override profile/layout if provided via CLI
        profile_path = args.profile.resolve() if args.profile else item.profile_path.resolve()
        layout_path = args.layout.resolve() if args.layout else item.layout_path.resolve()

        profile = load_semantic_profile(profile_path)
        layout_profile = load_layout_profile(layout_path)
        layout = (
            layout_profile.pages[item.page_num - 1]
            if item.page_num - 1 < len(layout_profile.pages)
            else layout_profile.pages[0]
        )

        expected_options = expand_ground_truth_labels(item, profile)
        if args.question:
            expected_options = [opt for opt in expected_options if opt.question_id == args.question]
            if not expected_options:
                continue

        try:
            if item.pdf_path.suffix.lower() == ".pdf":
                rendered = next(
                    (
                        p
                        for p in extract_pages(item.pdf_path, dpi=300)
                        if p.page_number == item.page_num
                    ),
                    None,
                )
                if not rendered:
                    raise ValueError(f"Page {item.page_num} not found in PDF {item.pdf_path}.")
            else:
                img = Image.open(item.pdf_path).convert("RGB")
                pdf_width_pt = img.width * 72.0 / 300.0
                pdf_height_pt = img.height * 72.0 / 300.0
                rendered = RenderedPage(
                    page_number=item.page_num,
                    width_px=img.width,
                    height_px=img.height,
                    pdf_width_pt=float(pdf_width_pt),
                    pdf_height_pt=float(pdf_height_pt),
                    image=img,
                )

            if args.reference_type == "golden":
                ref_path = golden_ref_path
            else:
                # hybrid not fully implemented in script, fallback to golden
                ref_path = golden_ref_path

            if not ref_path.exists():
                raise FileNotFoundError(f"Reference image {ref_path} not found.")

            reference_img = Image.open(ref_path).convert("RGB")

            aligned_page = align_page(rendered, reference_img, alignment_config)
            page_evidence = extract_mark_evidence(aligned_page, profile, layout, reference_img)
            mark_scores = evidence_to_mark_scores(page_evidence, aligned_page)
            normalized_result = route_page(mark_scores, profile, item.page_num, routing_config)

            result = evaluate_page(
                item.item_id,
                str(item.pdf_path),
                item.page_num,
                expected_options,
                normalized_result,
                page_evidence,
            )

            if args.debug_artifacts:
                debug_out_dir = run_dir / "debug" / item.item_id
                debug_report = run_pixel_probe_cli(
                    item.pdf_path,
                    item.page_num,
                    debug_out_dir,
                    profile_path,
                    ref_path,
                    args.question,
                    run_config["alignment"],
                    run_config["routing"],
                )
                if debug_report:
                    base_report_path = debug_report.relative_to(run_dir).as_posix()
                    base_report_dir = base_report_path.replace("index.html", "")
                    # pixel_probe generates an index.html and also per-question HTMLs
                    # (not implemented yet in this script, but structure is there)
                    for q in result.question_results:
                        q.debug_report_path = (
                            base_report_path  # The main HTML has anchors for questions
                        )
                        for o in q.option_results:
                            # In real pixel_probe we might have granular paths,
                            # but for now anchor to main report
                            o.debug_report_path = (
                                f"{base_report_path}#{q.question_id}_{o.option_id}"
                            )
                            o.evidence_path = f"{base_report_dir}{q.question_id}/{o.option_id}/09_overlay_decision.png"
                            if "Q14" in q.question_id:
                                o.q14_diagnostic_path = (
                                    f"{base_report_dir}Q14/{o.option_id}/q14_decision_overlay.png"
                                )

                    report_json_path = debug_out_dir / "report.json"
                    if report_json_path.exists():
                        with open(report_json_path, encoding="utf-8") as f:
                            r_data = json.load(f)
                            result.q14_diagnostics = r_data.get("q14_diagnostics", [])

            page_results.append(result)

        except Exception as e:
            import traceback

            traceback.print_exc()
            result = evaluate_page(
                item.item_id,
                str(item.pdf_path),
                item.page_num,
                expected_options,
                None,
                None,
                error_msg=str(e),
            )
            page_results.append(result)

            if args.fail_fast:
                break

    reporter = GroundTruthReporter(run_dir)
    reporter.write_report(run_id, run_config, page_results)


if __name__ == "__main__":
    main()
