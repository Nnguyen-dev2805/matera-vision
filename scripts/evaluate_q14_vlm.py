import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Import existing core utilities
from matera.core.profile import load_semantic_profile
from matera.core.q14_vlm_profile import load_q14_vlm_profile
from matera.data.contracts import RenderedPage
from matera.evaluation.ground_truth import expand_ground_truth_labels, load_ground_truth
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig
from matera.vision.q14_vlm_resolver import Q14VlmRuntime, resolve_q14_with_vlm
from matera.vlm.gemini_client import GeminiVlmClient
from matera.vlm.metrics import calculate_q14_vlm_metrics
from matera.vlm.models import Q14VlmOptionResult


def generate_index_html(run_dir: Path, metrics: dict, results: list[Q14VlmOptionResult]):
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head><title>Q14 VLM Evaluation Report</title>",
        "<style>",
        "body { font-family: sans-serif; background: #121212; color: #eee; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }",
        "th, td { border: 1px solid #444; padding: 8px; text-align: left; }",
        "th { background: #333; }",
        ".match { color: #4caf50; font-weight: bold; }",
        ".mismatch { color: #f44336; font-weight: bold; }",
        ".review { color: #ff9800; font-weight: bold; }",
        ".card { background: #222; padding: 15px; margin-bottom: 20px; border-radius: 5px; }",
        ".crop-img { max-width: 600px; height: auto; border: 1px solid #555; }",
        "</style></head><body>",
        "<h1>Q14 VLM Evaluation Report</h1>",
        "<h2>Summary Metrics</h2>",
        "<table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
    ]

    for k, v in metrics.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        html_lines.append(f"<tr><td>{k}</td><td>{v}</td></tr>")

    html_lines.append("</table>")

    # Group results by page
    from collections import defaultdict

    pages_map = defaultdict(list)
    for r in results:
        pages_map[r.page_number].append(r)

    for page_num in sorted(pages_map.keys()):
        page_results = pages_map[page_num]

        # We need the relative path of the image for html
        img_path = page_results[0].image_path
        if img_path:
            img_rel_path = Path(img_path).relative_to(run_dir).as_posix()
        else:
            img_rel_path = ""

        html_lines.append("<div class='card'>")
        html_lines.append(f"<h2>Page {page_num}</h2>")

        if img_rel_path:
            html_lines.append(f"<p><img class='crop-img' src='{img_rel_path}'></p>")

        html_lines.append("<table>")
        html_lines.append(
            (
                "<tr><th>Option ID</th><th>Expected</th><th>VLM Parsed</th>"
                "<th>Raw State</th><th>Reason</th><th>Error</th></tr>"
            )
        )

        for r in sorted(page_results, key=lambda x: x.option_id):
            exp = r.expected_state
            act = r.vlm_state

            css_class = "match"
            if act == "NEED_REVIEW":
                css_class = "review"
            elif exp != act:
                css_class = "mismatch"

            html_lines.append("<tr>")
            html_lines.append(f"<td>{r.option_id}</td>")
            html_lines.append(f"<td>{exp}</td>")
            html_lines.append(f"<td class='{css_class}'>{act}</td>")
            html_lines.append(f"<td>{r.raw_vlm_state or ''}</td>")
            html_lines.append(f"<td>{r.reason or ''}</td>")
            html_lines.append(f"<td>{r.parse_error or ''}</td>")
            html_lines.append("</tr>")

        html_lines.append("</table>")
        html_lines.append("</div>")

    html_lines.append("</body></html>")

    with open(run_dir / "index.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))


def get_aligned_page_from_debug(debug_root: Path, page_id: str) -> Image.Image | None:
    # page_id is like "page_001"
    # we search for *aligned*.png inside debug_root/debug/page_id/
    search_dir = debug_root / "debug" / page_id
    if not search_dir.exists():
        # Fallback to search recursively if structure is nested
        search_dir = debug_root

    for p in search_dir.rglob("*.png"):
        if "aligned" in p.name.lower() and page_id in p.as_posix():
            return Image.open(p).convert("RGB")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=str, default=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    )
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--page", type=int)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--profile", type=Path, default=Path("profiles/semantic.json"))
    parser.add_argument("--layout", type=Path, default=Path("profiles/layout.json"))
    parser.add_argument("--reference-type", type=str, default="golden")
    parser.add_argument("--use-existing-debug-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")

    args = parser.parse_args()

    dataset = load_ground_truth(args.ground_truth)
    profile = load_semantic_profile(args.profile)

    q14_profile_path = Path("profiles/q14_vlm.json")
    if not q14_profile_path.exists():
        print(f"Error: {q14_profile_path} not found.")
        sys.exit(1)
    q14_profile = load_q14_vlm_profile(q14_profile_path)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    vlm_client = None
    if not args.dry_run:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(
                f"Error: API key environment variable {args.api_key_env} not set.", file=sys.stderr
            )
            sys.exit(1)
        vlm_client = GeminiVlmClient(api_key=api_key)

    # Save run prompt config
    from matera.vision.q14_vlm_resolver import Q14_PROMPT

    with open(run_dir / "prompt.txt", "w", encoding="utf-8") as f:
        f.write(Q14_PROMPT)

    items = dataset.items
    if args.page is not None:
        items = [i for i in items if i.page_num == args.page]
    if args.max_pages is not None:
        items = items[: args.max_pages]

    golden_ref_path = args.profile.parent / "reference_template.png"
    if not golden_ref_path.exists():
        print(f"Error: Reference image {golden_ref_path} not found.")
        sys.exit(1)

    all_results = []

    latencies = []
    provider_error_pages = 0

    for item in items:
        page_id = item.item_id  # "page_001" etc.
        print(f"Processing {page_id}...")

        debug_dir = run_dir / "debug" / page_id
        debug_dir.mkdir(parents=True, exist_ok=True)

        aligned_img = None
        if args.use_existing_debug_root:
            aligned_img = get_aligned_page_from_debug(args.use_existing_debug_root, page_id)

        if aligned_img is None:
            # Fallback to normal rendering and alignment
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
            reference_img = Image.open(golden_ref_path).convert("RGB")
            alignment_config = AlignmentConfig(
                algorithm="orb", transform_model="affine", inlier_threshold=0.05
            )
            aligned_page = align_page(rendered, reference_img, alignment_config)
            aligned_img = aligned_page.image

        expected_options = expand_ground_truth_labels(item, profile)
        q14_expected = {
            opt.option_id: opt.expected_state
            for opt in expected_options
            if "Q14" in opt.question_id
        }

        runtime = Q14VlmRuntime(
            profile=q14_profile,
            client=vlm_client,
            model=args.model,
            timeout_s=60.0,
            debug_dir=debug_dir,
        )

        if args.dry_run:
            print(f"[Dry Run] Generated crops for {page_id}.")
            continue

        evidence = resolve_q14_with_vlm(aligned_img, runtime)

        if evidence.latency_ms is not None:
            latencies.append(evidence.latency_ms)

        if evidence.provider_error:
            provider_error_pages += 1
            print(f"Provider Error on {page_id}: {evidence.provider_error}", file=sys.stderr)
            if args.fail_fast:
                break

        for opt_ev in evidence.options:
            result = Q14VlmOptionResult(
                page_number=item.page_num,
                option_id=opt_ev.option_id,
                expected_state=q14_expected.get(opt_ev.option_id),
                vlm_state=opt_ev.decision,
                raw_vlm_state=opt_ev.raw_state,
                reason=opt_ev.reason,
                parse_error=opt_ev.parse_error,
                image_path=evidence.crop_path,
            )
            all_results.append(result)

    if args.dry_run:
        print("Dry run complete.")
        return

    metrics = calculate_q14_vlm_metrics(all_results)
    metrics["provider_error_pages"] = provider_error_pages
    metrics["mean_latency_ms"] = sum(latencies) / len(latencies) if latencies else None

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    generate_index_html(run_dir, metrics, all_results)

    with open(run_dir / "details.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "page_number",
                "option_id",
                "expected_state",
                "vlm_state",
                "raw_vlm_state",
                "is_true_positive",
                "is_true_negative",
                "is_false_positive",
                "is_false_negative",
                "is_need_review",
                "reason",
                "parse_error",
                "provider_error",
                "image_path",
            ]
        )
        for r in all_results:
            is_tp = r.expected_state == "MARKED" and r.vlm_state == "MARKED"
            is_tn = r.expected_state == "BLANK" and r.vlm_state == "BLANK"
            is_fp = r.expected_state == "BLANK" and r.vlm_state == "MARKED"
            is_fn = r.expected_state == "MARKED" and r.vlm_state == "BLANK"
            is_nr = r.vlm_state == "NEED_REVIEW"

            provider_err = (
                "Yes"
                if provider_error_pages > 0 and r.parse_error and "httpx" in r.parse_error.lower()
                else ""
            )

            writer.writerow(
                [
                    r.page_number,
                    r.option_id,
                    r.expected_state,
                    r.vlm_state,
                    r.raw_vlm_state,
                    is_tp,
                    is_tn,
                    is_fp,
                    is_fn,
                    is_nr,
                    r.reason,
                    r.parse_error,
                    provider_err,
                    r.image_path,
                ]
            )

    print(f"Evaluation complete. Reports generated in {run_dir}")


if __name__ == "__main__":
    main()
