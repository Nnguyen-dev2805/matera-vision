"""
Script tạo báo cáo debug trực quan cho V17.
Chạy pipeline trên toàn bộ 10 trang, lưu 3 ảnh/trang:
  1. Ảnh gốc đã nắn (aligned)
  2. Ảnh kết quả (vẽ khung màu + nhãn)
  3. Ảnh debug mask (mặt nạ mực sau lọc)
Xuất ra markdown report có nhúng hình.
"""
import sys
import json
import cv2
import pickle
import numpy as np
import math
from pathlib import Path
from PIL import Image
from collections import defaultdict
from scipy.spatial.distance import cdist
from skimage.feature import hog

from matera.data.extract import extract_pages
from matera.vision.alignment import align_page
from matera.vision.contracts import AlignmentConfig
from matera.core.layout import load_layout_profile

ARTIFACT_DIR = Path("C:/Users/ADMIN/.gemini/antigravity-ide/brain/365fd45d-bf9f-4026-9c6a-9c4817a79d98")
DEBUG_IMG_DIR = ARTIFACT_DIR / "debug_v17"
DEBUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGS ---
NUM_BINS = 72
DEGREES_PER_BIN = 360 / NUM_BINS
MIN_INK_PER_BIN = 2
MARKED_THRESHOLD_DEG = 180
BLANK_THRESHOLD_DEG = 90
LOCAL_PAD = 15
OUTER_RADIUS = 32
GLOBAL_PAD = 20
MERGE_THRESHOLD = 15.0
EXTREMES_REJECT_THRESHOLD = 50.0
CHECKBOX_INNER_MARGIN = 5
HSV_SATURATION_THRESHOLD = 40

MODEL_PATH = Path("models/shape_classifier.pkl")
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj: self.parent[ri] = rj

def extract_hog_features(img):
    img_resized = cv2.resize(img, (24, 24))
    return hog(img_resized, orientations=8, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, feature_vector=True)

def get_text_bounding_box(ref_crop_gray):
    _, thresh = cv2.threshold(ref_crop_gray, 200, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    h, w = ref_crop_gray.shape
    cx, cy = w / 2.0, h / 2.0
    min_dist = float('inf')
    best = None
    for i in range(1, num_labels):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1: continue
        if max(bw / float(bh), bh / float(bw)) > 4.0: continue
        d = math.sqrt((centroids[i][0] - cx)**2 + (centroids[i][1] - cy)**2)
        if d < min_dist:
            min_dist = d
            best = (x, y, bw, bh)
    return best if best else (int(cx - 7), int(cy - 7), 14, 14)

def get_local_roi_crops(img_pil, ref_bgr, bbox, pad):
    x1, y1 = max(0, bbox.x - pad), max(0, bbox.y - pad)
    x2, y2 = min(img_pil.width, bbox.x + bbox.w + pad), min(img_pil.height, bbox.y + bbox.h + pad)
    crop = cv2.cvtColor(np.array(img_pil.crop((x1, y1, x2, y2))), cv2.COLOR_RGB2BGR)
    ref = ref_bgr[y1:y2, x1:x2]
    tg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(rg, tg)
    _, mask = cv2.threshold(cv2.GaussianBlur(diff, (3,3), 0), 30, 255, cv2.THRESH_BINARY)
    return crop, mask, (x1, y1, x2, y2), rg

def get_local_roi_crops_hsv(img_pil, ref_bgr, bbox, pad):
    x1, y1 = max(0, bbox.x - pad), max(0, bbox.y - pad)
    x2, y2 = min(img_pil.width, bbox.x + bbox.w + pad), min(img_pil.height, bbox.y + bbox.h + pad)
    crop = cv2.cvtColor(np.array(img_pil.crop((x1, y1, x2, y2))), cv2.COLOR_RGB2BGR)
    ref = ref_bgr[y1:y2, x1:x2]
    tg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(rg, tg)
    _, mask_diff = cv2.threshold(cv2.GaussianBlur(diff, (3,3), 0), 30, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, mask_color = cv2.threshold(hsv[:,:,1], HSV_SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
    return crop, cv2.bitwise_and(mask_diff, mask_color), (x1, y1, x2, y2)

def run_global(img_pil, ref_bgr, rois, result_bgr):
    if len(rois) <= 1: return set()
    xs = [r.bbox.x for r in rois] + [r.bbox.x + r.bbox.w for r in rois]
    ys = [r.bbox.y for r in rois] + [r.bbox.y + r.bbox.h for r in rois]
    x1, y1 = max(0, min(xs) - GLOBAL_PAD), max(0, min(ys) - GLOBAL_PAD)
    x2, y2 = min(img_pil.width, max(xs) + GLOBAL_PAD), min(img_pil.height, max(ys) + GLOBAL_PAD)
    crop = cv2.cvtColor(np.array(img_pil.crop((x1, y1, x2, y2))), cv2.COLOR_RGB2BGR)
    ref = ref_bgr[y1:y2, x1:x2]
    diff = cv2.absdiff(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    _, mask = cv2.threshold(cv2.GaussianBlur(diff, (3,3), 0), 30, 255, cv2.THRESH_BINARY)
    h, w = mask.shape
    mask[:15, :] = 0; mask[h-15:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = {}
    for r in rois:
        centers[r.option_id] = ((r.bbox.x + r.bbox.w/2.0) - x1, (r.bbox.y + r.bbox.h/2.0) - y1)
    
    valid = [c for c in contours if cv2.contourArea(c) > 30 and not ((cv2.boundingRect(c)[3] > 100 and cv2.boundingRect(c)[2] < 25) or (cv2.boundingRect(c)[2] > 100 and cv2.boundingRect(c)[3] < 25))]
    n = len(valid)
    if n == 0: return set()
    
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            pts1, pts2 = valid[i].reshape(-1,2), valid[j].reshape(-1,2)
            if np.min(cdist(pts1, pts2)) <= MERGE_THRESHOLD:
                l1, r1 = pts1[np.argmin(pts1[:,0])], pts1[np.argmax(pts1[:,0])]
                l2, r2 = pts2[np.argmin(pts2[:,0])], pts2[np.argmax(pts2[:,0])]
                if not (np.linalg.norm(l1-l2) > EXTREMES_REJECT_THRESHOLD and np.linalg.norm(r1-r2) > EXTREMES_REJECT_THRESHOLD):
                    uf.union(i, j)
    
    clusters = defaultdict(list)
    for i in range(n): clusters[uf.find(i)].append(valid[i])
    
    marked = set()
    for _, cnts in clusters.items():
        pts = np.vstack(cnts)
        area = sum(cv2.contourArea(c) for c in cnts)
        hull = cv2.convexHull(pts)
        ha = cv2.contourArea(hull)
        if ha > 1000 and (area / ha if ha > 0 else 1) < 0.4:
            ho = hull.copy()
            for p in ho: p[0][0] += x1; p[0][1] += y1
            cv2.drawContours(result_bgr, [ho], 0, (0, 255, 255), 2)
            for oid, (cx, cy) in centers.items():
                if cv2.pointPolygonTest(hull, (cx, cy), False) >= 0:
                    marked.add(oid)
    return marked

def process_hsv_ai(img_pil, ref_bgr, roi, prefix):
    """img_pil should be a PIL Image, not AlignedPage"""
    actual_img = img_pil.image if hasattr(img_pil, 'image') else img_pil
    _, mask, _ = get_local_roi_crops_hsv(actual_img, ref_bgr, roi.bbox, 0)
    h, w = mask.shape
    safe = np.zeros_like(mask)
    m = CHECKBOX_INNER_MARGIN
    if h > 2*m and w > 2*m:
        safe[m:h-m, m:w-m] = mask[m:h-m, m:w-m]
    ink = np.sum(safe > 0)
    if ink < 20:
        return "BLANK", f"{prefix}_L2({ink}px)", safe
    elif ink > 200:
        return "MARKED", f"{prefix}_L2({ink}px)", safe
    else:
        feat = extract_hog_features(safe)
        prob = clf.predict_proba([feat])[0][1]
        if prob > 0.85: return "MARKED", f"{prefix}_AI({prob:.0%})", safe
        elif prob < 0.15: return "BLANK", f"{prefix}_AI({prob:.0%})", safe
        else: return "AMBIGUOUS", f"{prefix}_AI({prob:.0%})", safe

def map_opt_q14(opt_id):
    return {"1": "0", "2": "1", "3": "2", "4": "3"}.get(opt_id, opt_id)

def main():
    layout = load_layout_profile(Path("profiles/matera-pre/v1/layout.json"))
    ref_align = Image.open("data/pages/page_1.png")
    median_ref = Image.open("scratch/synthetic_median_reference.png")
    ref_bgr = cv2.cvtColor(np.array(median_ref), cv2.COLOR_RGB2BGR)
    
    with open("data/example/ground_truth.json", "r", encoding="utf-8") as f:
        gt_all = json.load(f)
    
    pages = list(extract_pages(Path("data/example/matera-example.pdf")))
    acfg = AlignmentConfig(algorithm="orb", transform_model="affine", inlier_threshold=0.05)
    
    all_results = []
    page_summaries = []
    
    for pidx, pimg in enumerate(pages):
        pnum = pidx + 1
        print(f"Processing Page {pnum}...")
        ap = align_page(pimg, ref_align, acfg)
        
        # Ảnh gốc đã nắn
        orig_rgb = np.array(ap.image)
        cv2.imwrite(str(DEBUG_IMG_DIR / f"p{pnum}_1_original.png"), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))
        
        result_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR).copy()
        debug_bgr = np.zeros_like(result_bgr)
        
        gt_page = gt_all.get(f"page_{pnum}", {}).get("expected", {})
        
        rois_by_q = defaultdict(list)
        for roi in layout.pages[0].rois:
            rois_by_q[roi.question_id].append(roi)
        
        # Global
        gm = {}
        for qid, rs in rois_by_q.items():
            gm[qid] = run_global(ap.image, ref_bgr, rs, debug_bgr) if "Q14" not in qid else set()
        
        page_results = []
        
        for roi in layout.pages[0].rois:
            gt_qid = roi.question_id.replace(".", "_")
            if gt_qid not in gt_page: continue
            oid = roi.option_id
            gt_oid = map_opt_q14(oid) if gt_qid == "Q14" else oid.replace("a","0").replace("b","1").replace("c","2").replace("d","3").replace("e","4").replace("f","5").replace("g","6")
            is_marked = gt_oid in gt_page[gt_qid]
            
            if "Q14" not in roi.question_id:
                if oid in gm[roi.question_id]:
                    pred, method = "MARKED", "GLOBAL_HULL"
                    _, mk, cc, _ = get_local_roi_crops(ap.image, ref_bgr, roi.bbox, LOCAL_PAD)
                    x1,y1,x2,y2 = cc
                    try: debug_bgr[y1:y2, x1:x2] = np.maximum(debug_bgr[y1:y2, x1:x2], cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR))
                    except: pass
                else:
                    _, mk, cc, rg = get_local_roi_crops(ap.image, ref_bgr, roi.bbox, LOCAL_PAD)
                    x1,y1,x2,y2 = cc
                    tb = get_text_bounding_box(rg)
                    bx, by, bw, bh = max(0,tb[0]-2), max(0,tb[1]-2), tb[2]+4, tb[3]+4
                    h, w = mk.shape
                    cx, cy = w/2.0, h/2.0
                    Y, X = np.ogrid[:h, :w]
                    outer = ((X-cx)**2 + (Y-cy)**2) > OUTER_RADIUS**2
                    core = np.zeros((h,w), dtype=bool)
                    core[by:min(h,by+bh), bx:min(w,bx+bw)] = True
                    mr = mk.copy(); mr[core] = 0; mr[outer] = 0
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                    mr = cv2.morphologyEx(cv2.dilate(mr, k, iterations=1), cv2.MORPH_CLOSE, k, iterations=1)
                    
                    ys, xs = np.where(mr > 0)
                    deg = 0
                    if len(xs) > 0:
                        ang = np.degrees(np.arctan2(ys.astype(float)-cy, xs.astype(float)-cx)) % 360
                        hc, _ = np.histogram(ang, bins=NUM_BINS, range=(0,360))
                        ab = (hc >= MIN_INK_PER_BIN).astype(int)
                        for i in range(NUM_BINS):
                            if ab[i]==0 and ab[(i-1)%NUM_BINS]==1 and ab[(i+1)%NUM_BINS]==1: ab[i]=1
                        deg = np.sum(ab) * DEGREES_PER_BIN
                    
                    if deg >= MARKED_THRESHOLD_DEG:
                        pred, method = "MARKED", f"LOCAL_RADIAL({deg:.0f}°)"
                    elif deg <= BLANK_THRESHOLD_DEG:
                        pred, method = "BLANK", f"LOCAL_RADIAL({deg:.0f}°)"
                    else:
                        pred, method, mr = process_hsv_ai(ap, ref_bgr, roi, "FALLBACK")
                    
                    # Vẽ debug mask
                    dbg_color = cv2.cvtColor(mr, cv2.COLOR_GRAY2BGR)
                    try: debug_bgr[y1:y2, x1:x2] = np.maximum(debug_bgr[y1:y2, x1:x2], dbg_color)
                    except: pass
            else:
                pred, method, safe_mask = process_hsv_ai(ap, ref_bgr, roi, "Q14")
                bx1, by1 = max(0, roi.bbox.x), max(0, roi.bbox.y)
                bx2, by2 = bx1 + roi.bbox.w, by1 + roi.bbox.h
                dbg_c = cv2.cvtColor(safe_mask, cv2.COLOR_GRAY2BGR)
                try: debug_bgr[by1:by2, bx1:bx2] = np.maximum(debug_bgr[by1:by2, bx1:bx2], dbg_c)
                except: pass
            
            # Vẽ khung kết quả lên result_bgr
            rx, ry, rw, rh = roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h
            if pred == "MARKED":
                color = (0, 200, 0)  # Xanh lá
                label = "V"
            elif pred == "BLANK":
                color = (180, 180, 180)  # Xám nhạt
                label = ""
            else:
                color = (0, 255, 255)  # Vàng
                label = "?"
            
            cv2.rectangle(result_bgr, (rx, ry), (rx+rw, ry+rh), color, 2)
            if label:
                cv2.putText(result_bgr, label, (rx+2, ry+rh-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            status = "TP" if pred=="MARKED" and is_marked else \
                     "TN" if pred=="BLANK" and not is_marked else \
                     "FP" if pred=="MARKED" and not is_marked else \
                     "FN" if pred=="BLANK" and is_marked else \
                     f"AMBIG"
            
            # Vẽ thêm dấu đỏ cho FP/FN
            if status == "FP":
                cv2.rectangle(result_bgr, (rx-1, ry-1), (rx+rw+1, ry+rh+1), (0, 0, 255), 3)
                cv2.putText(result_bgr, "FP", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
            elif status == "FN":
                cv2.rectangle(result_bgr, (rx-1, ry-1), (rx+rw+1, ry+rh+1), (255, 0, 255), 3)
                cv2.putText(result_bgr, "FN", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)
            elif status == "AMBIG":
                cv2.rectangle(result_bgr, (rx-1, ry-1), (rx+rw+1, ry+rh+1), (0, 200, 255), 3)
                cv2.putText(result_bgr, "?!", (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 2)
            
            page_results.append({
                "q": roi.question_id, "opt": oid, "pred": pred,
                "gt": "MARKED" if is_marked else "BLANK",
                "status": status, "method": method
            })
        
        # Lưu ảnh
        cv2.imwrite(str(DEBUG_IMG_DIR / f"p{pnum}_2_result.png"), result_bgr)
        
        # Tô màu debug mask cho dễ nhìn
        debug_colored = debug_bgr.copy()
        # Vẽ ROI outlines lên debug
        for roi in layout.pages[0].rois:
            rx, ry, rw, rh = roi.bbox.x, roi.bbox.y, roi.bbox.w, roi.bbox.h
            cv2.rectangle(debug_colored, (rx, ry), (rx+rw, ry+rh), (50, 50, 50), 1)
        cv2.imwrite(str(DEBUG_IMG_DIR / f"p{pnum}_3_debug.png"), debug_colored)
        
        # Thống kê trang
        tp = sum(1 for r in page_results if r["status"] == "TP")
        tn = sum(1 for r in page_results if r["status"] == "TN")
        fp = sum(1 for r in page_results if r["status"] == "FP")
        fn = sum(1 for r in page_results if r["status"] == "FN")
        amb = sum(1 for r in page_results if r["status"] == "AMBIG")
        total = len(page_results)
        acc = (tp + tn) / total * 100 if total > 0 else 0
        
        page_summaries.append({
            "page": pnum, "total": total, "tp": tp, "tn": tn,
            "fp": fp, "fn": fn, "amb": amb, "acc": acc,
            "errors": [r for r in page_results if r["status"] not in ["TP", "TN"]]
        })
        all_results.extend(page_results)
    
    # === TẠO BÁO CÁO MARKDOWN ===
    md = []
    md.append("# 🛡️ Báo Cáo Debug Trực Quan — Matera Vision V17")
    md.append("")
    md.append("**Kiến trúc:** Phòng thủ nhiều lớp (Global Hull → Local Radial → Pixel Count → HOG+SVM)")
    md.append(f"**Bộ dữ liệu:** 10 trang scan, tổng cộng {len(all_results)} ô đáp án")
    md.append("")
    
    # Tổng kết
    total_tp = sum(s["tp"] for s in page_summaries)
    total_tn = sum(s["tn"] for s in page_summaries)
    total_fp = sum(s["fp"] for s in page_summaries)
    total_fn = sum(s["fn"] for s in page_summaries)
    total_amb = sum(s["amb"] for s in page_summaries)
    total_all = len(all_results)
    total_acc = (total_tp + total_tn) / total_all * 100
    
    md.append("## Tổng kết Toàn hệ thống")
    md.append("")
    md.append("| Chỉ số | Giá trị |")
    md.append("|--------|---------|")
    md.append(f"| Tổng ô | **{total_all}** |")
    md.append(f"| ✅ True Positive | {total_tp} |")
    md.append(f"| ⬜ True Negative | {total_tn} |")
    md.append(f"| 🔴 False Positive | **{total_fp}** |")
    md.append(f"| 🟣 False Negative | **{total_fn}** |")
    md.append(f"| 🟡 Ambiguous (chờ review) | {total_amb} |")
    md.append(f"| **Accuracy** | **{total_acc:.2f}%** |")
    md.append("")
    
    md.append("> **Chú thích hình ảnh:**")
    md.append("> - 🟢 Khung xanh lá + chữ **V** = Hệ thống phát hiện ĐÃ ĐÁNH DẤU")
    md.append("> - ⬜ Khung xám = Hệ thống phát hiện TRỐNG")
    md.append("> - 🔴 Khung đỏ đậm + **FP** = Bắt nhầm (False Positive)")
    md.append("> - 🟣 Khung tím + **FN** = Bỏ sót (False Negative)")
    md.append("> - 🟡 Khung vàng + **?!** = Mập mờ, chờ người duyệt (Ambiguous)")
    md.append("> - 🟨 Đường viền vàng trên debug = Convex Hull từ Global Topology")
    md.append("")
    md.append("---")
    
    for s in page_summaries:
        pn = s["page"]
        md.append(f"")
        md.append(f"## Trang {pn}")
        md.append(f"")
        md.append(f"**Accuracy: {s['acc']:.1f}%** — TP: {s['tp']} | TN: {s['tn']} | FP: {s['fp']} | FN: {s['fn']} | Ambig: {s['amb']}")
        md.append("")
        
        # Carousel: Original -> Result -> Debug
        md.append("````carousel")
        md.append(f"**1. Ảnh Gốc (Đã nắn thẳng)**")
        md.append(f"![Trang {pn} — Gốc]({(DEBUG_IMG_DIR / f'p{pn}_1_original.png').as_posix()})")
        md.append("<!-- slide -->")
        md.append(f"**2. Kết Quả Nhận Dạng**")
        md.append(f"![Trang {pn} — Kết quả]({(DEBUG_IMG_DIR / f'p{pn}_2_result.png').as_posix()})")
        md.append("<!-- slide -->")
        md.append(f"**3. Debug Mask (X-quang mực)**")
        md.append(f"![Trang {pn} — Debug]({(DEBUG_IMG_DIR / f'p{pn}_3_debug.png').as_posix()})")
        md.append("````")
        md.append("")
        
        if s["errors"]:
            md.append(f"**Các ô bị lỗi / mập mờ tại Trang {pn}:**")
            md.append("")
            md.append("| Câu | Lựa chọn | Dự đoán | Ground Truth | Trạng thái | Phương pháp |")
            md.append("|-----|----------|---------|--------------|------------|-------------|")
            for e in s["errors"]:
                md.append(f"| {e['q']} | {e['opt']} | {e['pred']} | {e['gt']} | **{e['status']}** | {e['method']} |")
            md.append("")
        else:
            md.append(f"> ✅ **Trang {pn}: HOÀN HẢO — Không có lỗi nào!**")
            md.append("")
        
        md.append("---")
    
    report_path = ARTIFACT_DIR / "v17_debug_visual_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    
    print(f"DONE! Report saved to {report_path}")

if __name__ == "__main__":
    main()
