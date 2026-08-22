import json
from dataclasses import dataclass
from pathlib import Path

from matera.core.errors import ProfileValidationError
from matera.core.profile import FormProfile


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if type(self.x) is not int or type(self.y) is not int:
            raise ValueError("x and y must be ints")
        if type(self.w) is not int or type(self.h) is not int:
            raise ValueError("w and h must be ints")
        if self.w <= 0 or self.h <= 0:
            raise ValueError("w and h must be positive ints")


@dataclass(frozen=True)
class AnchorDef:
    anchor_id: str
    anchor_type: str
    bbox: BoundingBox

    def __post_init__(self) -> None:
        if not self.anchor_id or not isinstance(self.anchor_id, str):
            raise ValueError("anchor_id must be a non-empty string")
        if not self.anchor_type or not isinstance(self.anchor_type, str):
            raise ValueError("anchor_type must be a non-empty string")
        if not isinstance(self.bbox, BoundingBox):
            raise ValueError("bbox must be a BoundingBox")


@dataclass(frozen=True)
class RoiDef:
    question_id: str
    option_id: str
    bbox: BoundingBox
    mark_strategy_override: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id or not isinstance(self.question_id, str):
            raise ValueError("question_id must be a non-empty string")
        if not self.option_id or not isinstance(self.option_id, str):
            raise ValueError("option_id must be a non-empty string")
        if not isinstance(self.bbox, BoundingBox):
            raise ValueError("bbox must be a BoundingBox")
        if self.mark_strategy_override not in {"circle", "checkbox", "rating", None}:
            raise ValueError(
                "mark_strategy_override must be one of: circle, checkbox, rating, or None"
            )


@dataclass(frozen=True)
class PageLayout:
    page_number: int
    width_px: int
    height_px: int
    anchors: tuple[AnchorDef, ...]
    rois: tuple[RoiDef, ...]

    def __post_init__(self) -> None:
        if type(self.page_number) is not int or self.page_number < 1:
            raise ValueError("page_number must be a positive int")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise ValueError("width_px must be a positive int")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise ValueError("height_px must be a positive int")

        if not isinstance(self.anchors, tuple):
            raise ValueError("anchors must be a tuple")
        for a in self.anchors:
            if not isinstance(a, AnchorDef):
                raise ValueError("All anchors must be AnchorDef instances")
            if a.bbox.x < 0 or a.bbox.y < 0:
                raise ValueError("anchor bbox must have non-negative x/y")
            if a.bbox.x + a.bbox.w > self.width_px or a.bbox.y + a.bbox.h > self.height_px:
                raise ValueError("anchor bbox exceeds page dimensions")

        if not isinstance(self.rois, tuple):
            raise ValueError("rois must be a tuple")
        for r in self.rois:
            if not isinstance(r, RoiDef):
                raise ValueError("All rois must be RoiDef instances")
            if r.bbox.x < 0 or r.bbox.y < 0:
                raise ValueError("roi bbox must have non-negative x/y")
            if r.bbox.x + r.bbox.w > self.width_px or r.bbox.y + r.bbox.h > self.height_px:
                raise ValueError("roi bbox exceeds page dimensions")


@dataclass(frozen=True)
class LayoutProfile:
    form_id: str
    form_version: str
    coordinate_space: str
    reference_dpi: int
    pages: tuple[PageLayout, ...]

    def __post_init__(self) -> None:
        if not self.form_id or not isinstance(self.form_id, str):
            raise ValueError("form_id must be a non-empty string")
        if not self.form_version or not isinstance(self.form_version, str):
            raise ValueError("form_version must be a non-empty string")
        if self.coordinate_space != "absolute_pixel":
            raise ValueError("coordinate_space must be 'absolute_pixel'")
        if type(self.reference_dpi) is not int or self.reference_dpi <= 0:
            raise ValueError("reference_dpi must be a positive int")

        if not isinstance(self.pages, tuple):
            raise ValueError("pages must be a tuple")
        if not self.pages:
            raise ValueError("pages cannot be empty")

        for p in self.pages:
            if not isinstance(p, PageLayout):
                raise ValueError("All pages must be PageLayout instances")

        page_nums = [p.page_number for p in self.pages]
        if sorted(page_nums) != list(range(1, len(self.pages) + 1)):
            raise ValueError("page numbers must be sequential starting from 1")


def _raise_error(path: str, field: str, code: str, reason: str) -> None:
    raise ProfileValidationError(path, field, code, reason)


def _parse_bbox(path_str: str, field_path: str, data: dict) -> BoundingBox:
    if not isinstance(data, dict):
        _raise_error(path_str, field_path, "INVALID_TYPE", "must be an object")
    x = data.get("x")
    y = data.get("y")
    w = data.get("w")
    h = data.get("h")
    try:
        return BoundingBox(x, y, w, h)  # type: ignore
    except ValueError as e:
        _raise_error(path_str, field_path, "INVALID_FIELD", str(e))
        raise  # unreachable


def load_layout_profile(path: Path, semantic: FormProfile | None = None) -> LayoutProfile:
    """Loads, strictly validates, and cross-checks a layout profile JSON."""
    path_str = str(path)
    if not path.exists():
        _raise_error(path_str, "", "FILE_NOT_FOUND", "Profile file does not exist")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        _raise_error(path_str, "", "INVALID_JSON", str(e))

    if not isinstance(data, dict):
        _raise_error(path_str, "", "INVALID_ROOT", "Root must be a JSON object")

    form_id = data.get("form_id")
    form_version = data.get("form_version")
    coord_space = data.get("coordinate_space")
    ref_dpi = data.get("reference_dpi")

    if semantic is not None:
        if form_id != semantic.form_id:
            _raise_error(path_str, "form_id", "SEMANTIC_MISMATCH", f"Expected {semantic.form_id}")
        if form_version != semantic.form_version:
            _raise_error(
                path_str, "form_version", "SEMANTIC_MISMATCH", f"Expected {semantic.form_version}"
            )

    raw_pages = data.get("pages")
    if not isinstance(raw_pages, list):
        _raise_error(path_str, "pages", "INVALID_TYPE", "must be a list")

    pages = []
    seen_rois: set[tuple[int, str, str]] = set()

    for i, p_data in enumerate(raw_pages):
        field_p = f"pages[{i}]"
        if not isinstance(p_data, dict):
            _raise_error(path_str, field_p, "INVALID_TYPE", "must be an object")

        page_num = p_data.get("page_number")
        width_px = p_data.get("width_px")
        height_px = p_data.get("height_px")

        raw_anchors = p_data.get("anchors")
        if not isinstance(raw_anchors, list):
            _raise_error(path_str, f"{field_p}.anchors", "INVALID_TYPE", "must be a list")

        anchors = []
        for j, a_data in enumerate(raw_anchors):
            field_a = f"{field_p}.anchors[{j}]"
            if not isinstance(a_data, dict):
                _raise_error(path_str, field_a, "INVALID_TYPE", "must be an object")

            bbox = _parse_bbox(path_str, f"{field_a}.bbox", a_data.get("bbox", {}))
            try:
                anchors.append(
                    AnchorDef(
                        anchor_id=a_data.get("anchor_id"),  # type: ignore
                        anchor_type=a_data.get("anchor_type"),  # type: ignore
                        bbox=bbox,
                    )
                )
            except ValueError as e:
                _raise_error(path_str, field_a, "INVALID_FIELD", str(e))

        raw_rois = p_data.get("rois")
        if not isinstance(raw_rois, list):
            _raise_error(path_str, f"{field_p}.rois", "INVALID_TYPE", "must be a list")

        rois = []
        for j, r_data in enumerate(raw_rois):
            field_r = f"{field_p}.rois[{j}]"
            if not isinstance(r_data, dict):
                _raise_error(path_str, field_r, "INVALID_TYPE", "must be an object")

            q_id = r_data.get("question_id")
            opt_id = r_data.get("option_id")
            bbox = _parse_bbox(path_str, f"{field_r}.bbox", r_data.get("bbox", {}))

            roi_key = (page_num, q_id, opt_id)
            if roi_key in seen_rois:
                _raise_error(path_str, field_r, "DUPLICATE_ROI", "ROI already defined")
            seen_rois.add(roi_key)  # type: ignore

            if semantic is not None:
                # Check semantic existence
                q_def = next((q for q in semantic.questions if q.question_id == q_id), None)
                if not q_def:
                    _raise_error(
                        path_str,
                        f"{field_r}.question_id",
                        "MISSING_SEMANTIC_REF",
                        "Question ID not in semantic profile",
                    )

                opt_def = next((o for o in q_def.options if o.option_id == opt_id), None)
                if not opt_def:
                    _raise_error(
                        path_str,
                        f"{field_r}.option_id",
                        "MISSING_SEMANTIC_REF",
                        "Option ID not in semantic profile",
                    )

            try:
                rois.append(
                    RoiDef(
                        question_id=q_id,  # type: ignore
                        option_id=opt_id,  # type: ignore
                        bbox=bbox,
                        mark_strategy_override=r_data.get("mark_strategy_override"),
                    )
                )
            except ValueError as e:
                _raise_error(path_str, field_r, "INVALID_FIELD", str(e))

        try:
            pages.append(
                PageLayout(
                    page_number=page_num,  # type: ignore
                    width_px=width_px,  # type: ignore
                    height_px=height_px,  # type: ignore
                    anchors=tuple(anchors),
                    rois=tuple(rois),
                )
            )
        except ValueError as e:
            _raise_error(path_str, field_p, "INVALID_FIELD", str(e))

    if semantic is not None:
        # Check for orphaned options per page (every page must map all semantic options)
        semantic_keys = set()
        for q in semantic.questions:
            for o in q.options:
                semantic_keys.add((q.question_id, o.option_id))

        for page in pages:
            page_layout_keys = {(r.question_id, r.option_id) for r in page.rois}
            orphans = semantic_keys - page_layout_keys
            if orphans:
                _raise_error(
                    path_str,
                    f"pages[{page.page_number - 1}].rois",
                    "ORPHANED_OPTION",
                    f"Options missing ROIs on page {page.page_number}: {orphans}",
                )

    try:
        return LayoutProfile(
            form_id=form_id,  # type: ignore
            form_version=form_version,  # type: ignore
            coordinate_space=coord_space,  # type: ignore
            reference_dpi=ref_dpi,  # type: ignore
            pages=tuple(pages),
        )
    except ValueError as e:
        _raise_error(path_str, "", "INVALID_FIELD", str(e))
        raise  # unreachable
