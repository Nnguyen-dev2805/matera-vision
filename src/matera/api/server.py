import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from matera.core.layout import load_layout_profile
from matera.core.profile import load_semantic_profile
from matera.data.extract import extract_pages
from matera.export.excel import export_to_excel
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig, RoutingConfig
from matera.vision.mark import extract_mark_scores
from matera.vision.routing import route_page

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("matera-api")

app = FastAPI(
    title="Matera OMR Web API",
    description="Backend API for Matera Vision OMR Processing System",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(".tmp_uploads", exist_ok=True)
app.mount("/api/debug-images", StaticFiles(directory=".tmp_uploads"), name="debug-images")



def _process_single_file_worker(
    pdf_path: Path,
    file_idx: int,
    total_files: int,
    profile_dir: Path,
    mp_queue: Any,
    session_id: str,
    debug: bool = False,
) -> tuple[int, int, list]:
    """Worker function to process a single PDF file in a separate process."""
    semantic_path = profile_dir / "semantic.json"
    layout_path = profile_dir / "layout.json"

    if not semantic_path.exists() or not layout_path.exists():
        logger.error(f"Config profiles not found in {profile_dir}")
        return 0, 0, []

    semantic_profile = load_semantic_profile(semantic_path)
    layout_profile = load_layout_profile(layout_path)

    ref_img_path = Path("scratch/synthetic_median_reference.png")
    if not ref_img_path.exists():
        reference_img = Image.new("RGB", (2480, 3508), color=(255, 255, 255))
    else:
        reference_img = Image.open(ref_img_path).convert("RGB")

    alignment_config = AlignmentConfig(
        algorithm="orb",
        transform_model="affine",
        inlier_threshold=0.05,
    )
    routing_config = RoutingConfig(low_threshold=0.2, high_threshold=0.6)
    page_layout = layout_profile.pages[0]

    rel_name = pdf_path.name
    mp_queue.put({
        "type": "file_start",
        "file_index": file_idx,
        "total_files": total_files,
        "file_name": rel_name,
        "message": f"Loading file [{file_idx}/{total_files}]: {rel_name}",
        "percent": int(((file_idx - 1) / total_files) * 100),
    })

    file_results = []
    total_review_count = 0
    total_selected_count = 0

    try:
        pages = list(extract_pages(pdf_path))
    except Exception as e:
        logger.error(f"Error reading file {pdf_path}: {e}")
        return 0, 0, []

    num_pages = len(pages)

    for p_idx, page in enumerate(pages, start=1):
        t_start = time.time()
        page_percent = int(((file_idx - 1 + (p_idx / max(num_pages, 1))) / total_files) * 100)

        try:
            aligned_page = align_page(page, reference_img, alignment_config)
            mark_scores = extract_mark_scores(
                aligned_page, semantic_profile, page_layout, reference_img
            )
            result = route_page(mark_scores, semantic_profile, page.page_number, routing_config)
            
            import dataclasses
            result = dataclasses.replace(result, file_name=rel_name)
            file_results.append(result)
            
            selected_answers = [
                f"{a.answer_key.question_id}_{a.answer_key.option_id}"
                for a in result.answers
                if a.selected is True
            ]
            selected_count = len(selected_answers)
            total_selected_count += selected_count
            review_count = len(result.review_tasks)
            total_review_count += review_count
            t_elapsed = round(time.time() - t_start, 2)
            
            debug_images = []
            if debug:
                import cv2
                import numpy as np
                
                # 1. Original
                orig_bgr = cv2.cvtColor(np.array(aligned_page.image), cv2.COLOR_RGB2BGR)
                
                # 2. Ink Mask
                ref_bgr = cv2.cvtColor(np.array(reference_img), cv2.COLOR_RGB2BGR)
                orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
                ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(ref_gray, orig_gray)
                blurred = cv2.GaussianBlur(diff, (3, 3), 0)
                _, ink_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
                
                # 3. AI Overlay
                debug_full_mask = np.zeros_like(np.array(aligned_page.image))
                scores = extract_mark_scores(
                    aligned_page, semantic_profile, page_layout, reference_img,
                    debug_full_mask=debug_full_mask
                )
                overlay = cv2.addWeighted(orig_bgr, 0.6, debug_full_mask, 0.4, 0)
                
                roi_map = {(r.question_id, r.option_id): r for r in page_layout.rois}
                for s in scores:
                    roi = roi_map.get((s.question_id, s.option_id))
                    if not roi: continue
                    x, y, w, h = roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h
                    
                    if s.score == 1.0:
                        color = (0, 255, 0) # Green (BGR)
                    elif s.score == 0.0:
                        color = (0, 0, 255) # Red (BGR)
                    else:
                        color = (0, 255, 255) # Yellow (BGR)
                        
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                    
                    method_str = s.method.upper()
                    if "GLOBAL" in method_str:
                        label = "G"
                    elif "LOCAL" in method_str:
                        label = "L"
                    elif "AI" in method_str:
                        label = "A"
                    else:
                        label = "?"
                        
                    cv2.putText(overlay, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                
                debug_dir = Path(".tmp_uploads") / session_id / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Use safe filename without unicode to avoid cv2.imwrite issues on Windows
                base_name = f"file_{file_idx}_page_{page.page_number}"
                
                f_orig = f"{base_name}_1_original.png"
                f_ink = f"{base_name}_2_ink.png"
                f_overlay = f"{base_name}_3_overlay.png"
                
                # Use cv2.imencode + tofile for proper unicode path support (or just safe paths)
                # Since we use safe ascii paths, cv2.imwrite works fine.
                cv2.imwrite(str(debug_dir / f_orig), orig_bgr)
                cv2.imwrite(str(debug_dir / f_ink), ink_mask)
                cv2.imwrite(str(debug_dir / f_overlay), overlay)
                
                debug_images = [
                    f"/api/debug-images/{session_id}/debug/{f_orig}",
                    f"/api/debug-images/{session_id}/debug/{f_ink}",
                    f"/api/debug-images/{session_id}/debug/{f_overlay}"
                ]

            mp_queue.put({
                "type": "page_done",
                "file_index": file_idx,
                "total_files": total_files,
                "file_name": rel_name,
                "page_number": page.page_number,
                "total_pages_in_file": num_pages,
                "selected_count": selected_count,
                "selected_summary": ", ".join(selected_answers[:8]) + ("..." if selected_count > 8 else ""),
                "review_count": review_count,
                "status": "needs_review" if review_count > 0 else "resolved",
                "elapsed_seconds": t_elapsed,
                "percent": page_percent,
                "debug_images": debug_images,
                "message": f"[{file_idx}/{total_files}] {rel_name} - Page {page.page_number}: {selected_count} marks, {review_count} reviews ({t_elapsed}s)",
            })

        except Exception as ex:
            logger.error(f"Error processing page {page.page_number} of file {rel_name}: {ex}")
            mp_queue.put({
                "type": "page_error",
                "file_name": rel_name,
                "page_number": page.page_number,
                "error": str(ex),
            })

    return total_selected_count, total_review_count, file_results


def run_pipeline_on_files(
    pdf_files: List[Path],
    output_excel_path: Path,
    profile_dir: Path = Path("profiles"),
    progress_callback: Optional[Callable[[dict], None]] = None,
    session_id: str = "default_session",
    debug: bool = False,
) -> dict:
    """Execute V17 OMR pipeline concurrently on multiple PDF files and export to Excel."""
    semantic_path = profile_dir / "semantic.json"
    if not semantic_path.exists():
        raise RuntimeError(f"Config profiles not found in {profile_dir}")
    semantic_profile = load_semantic_profile(semantic_path)

    total_files = len(pdf_files)
    total_pages_count = 0
    total_review_count = 0
    total_selected_count = 0
    all_page_results = []

    if progress_callback:
        progress_callback({
            "type": "init",
            "total_files": total_files,
            "message": f"Found {total_files} PDF files. Initializing Multiprocessing V17 Engine...",
            "percent": 0,
        })

    manager = multiprocessing.Manager()
    mp_queue = manager.Queue()

    def queue_reader():
        while True:
            msg = mp_queue.get()
            if msg == "DONE":
                break
            if progress_callback:
                progress_callback(msg)

    reader_thread = threading.Thread(target=queue_reader, daemon=True)
    reader_thread.start()

    try:
        workers = max(1, os.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for file_idx, pdf_path in enumerate(pdf_files, start=1):
                futures.append(
                    executor.submit(
                        _process_single_file_worker,
                        pdf_path,
                        file_idx,
                        total_files,
                        profile_dir,
                        mp_queue,
                        session_id,
                        debug,
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    selected, reviews, results = future.result()
                    total_selected_count += selected
                    total_review_count += reviews
                    all_page_results.extend(results)
                    total_pages_count += len(results)
                except Exception as e:
                    logger.error(f"Worker process failed: {e}")
    finally:
        mp_queue.put("DONE")
        reader_thread.join(timeout=5.0)

    if not all_page_results:
        raise RuntimeError("No PDF pages were processed successfully.")

    # Ensure stable ordering of results by file_name and page_number
    all_page_results.sort(key=lambda r: (getattr(r, "file_name", "") or "", r.page_number))

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_excel(all_page_results, semantic_profile, output_excel_path)

    stats = {
        "total_files": total_files,
        "total_pages": total_pages_count,
        "total_answers": len(all_page_results) * sum(len(q.options) for q in semantic_profile.questions),
        "needs_review_count": total_review_count,
        "excel_path": str(output_excel_path),
        "excel_file_name": output_excel_path.name,
    }

    if progress_callback:
        progress_callback({
            "type": "complete",
            "percent": 100,
            "message": f"Processing complete. Extracted {total_pages_count} pages to Excel.",
            "stats": stats,
        })

    return stats



@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Matera Vision API is running"}


@app.post("/api/upload-session")
async def create_upload_session(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
):
    """Uploads PDF files from the browser native folder picker to a temporary batch folder."""
    if not session_id:
        session_id = str(uuid.uuid4())

    temp_dir = Path(".tmp_uploads") / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for f in files:
        if f.filename.lower().endswith(".pdf"):
            dest_file = temp_dir / Path(f.filename).name
            with open(dest_file, "wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
            saved_files.append(dest_file)

    return {
        "session_id": session_id,
        "total_files": len(saved_files),
        "files": [f.name for f in saved_files]
    }


@app.get("/api/process-stream")
async def process_stream(
    folder_path: Optional[str] = Query(None, description="Local folder path"),
    session_id: Optional[str] = Query(None, description="Uploaded session ID"),
    profile_dir: str = Query("profiles"),
    output_dir: str = Query("data/test"),
    debug: bool = Query(False, description="Enable visual debug output"),
):
    """Server-Sent Events (SSE) endpoint to stream real-time progress to frontend."""
    async def event_generator() -> AsyncGenerator[str, None]:
        # Determine source folder
        if session_id:
            folder = Path(".tmp_uploads") / session_id
        elif folder_path:
            folder = Path(folder_path)
        else:
            err_data = json.dumps({"type": "error", "message": "No folder or session specified"}, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        if not folder.exists() or not folder.is_dir():
            err_data = json.dumps({
                "type": "error",
                "message": f"Folder not found: {folder}"
            }, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        pdf_files = sorted(
            list(set(list(folder.rglob("*.pdf")) + list(folder.rglob("*.PDF"))))
        )

        if not pdf_files:
            err_data = json.dumps({
                "type": "error",
                "message": f"No PDF files found in folder: {folder}"
            }, ensure_ascii=False)
            yield f"data: {err_data}\n\n"
            return

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def queue_callback(event_dict: dict):
            loop.call_soon_threadsafe(queue.put_nowait, event_dict)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir)
        out_excel = out_dir / f"matera_results_{timestamp}.xlsx"

        # Run pipeline in background thread
        future = loop.run_in_executor(
            None,
            lambda: run_pipeline_on_files(
                pdf_files,
                out_excel,
                profile_dir=Path(profile_dir),
                progress_callback=queue_callback,
                session_id=session_id or "default",
                debug=debug,
            )
        )

        while not future.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.2)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue

        try:
            future.result()
        except Exception as e:
            logger.exception("Pipeline error during stream")
            err_data = json.dumps({
                "type": "error",
                "message": f"Error during processing: {str(e)}"
            }, ensure_ascii=False)
            yield f"data: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/download")
def download_excel(file_path: str = Query(..., description="Path to excel file")):
    """Download generated Excel result file."""
    path = Path(file_path).resolve()
    
    workspace_dir = Path.cwd().resolve()
    data_dir = (workspace_dir / "data").resolve()
    tmp_dir = (workspace_dir / ".tmp_uploads").resolve()
    
    if not (path.is_relative_to(data_dir) or path.is_relative_to(tmp_dir)):
        raise HTTPException(status_code=403, detail="Access denied.")
        
    if not path.exists():
        raise HTTPException(status_code=404, detail="File does not exist.")
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# Mount UI static files if directory exists
ui_dir = Path("frontend")
if ui_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")
