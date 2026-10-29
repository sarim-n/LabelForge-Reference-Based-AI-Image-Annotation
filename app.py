"""LabelForge - AI-Powered Image Auto-Labeling Studio.

Phase 4: Dataset Export, Validation, and Quality Checks.
Validates images and annotations, provides configurable reproducible train/val splits,
reports review approval rate and benchmark quality metrics (Precision/Recall/F1),
measures automated detection timing, and exports complete standard YOLO dataset archives.
"""

from collections import defaultdict
import inspect
import io
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import zipfile

from PIL import Image, ImageDraw
import streamlit as st

# Ensure Streamlit components v2 runtime manager discovers streamlit_drawable_canvas assets
try:
    from streamlit.runtime import Runtime
    import streamlit.components.v2.get_bidi_component_manager as _gbm_mod
    import streamlit.components.v2 as _v2_mod
    from streamlit.components.v2.component_manager import BidiComponentManager

    _local_bidi_mgr = None

    def _get_bidi_mgr():
        global _local_bidi_mgr
        if Runtime.exists():
            return Runtime.instance().bidi_component_registry
        if _local_bidi_mgr is None:
            _local_bidi_mgr = BidiComponentManager()
            _local_bidi_mgr.discover_and_register_components()
        return _local_bidi_mgr

    _gbm_mod.get_bidi_component_manager = _get_bidi_mgr
    _v2_mod.get_bidi_component_manager = _get_bidi_mgr
except Exception:
    pass

try:
    import streamlit_drawable_canvas as _sdc
    from streamlit_drawable_canvas import boxes_to_drawing, st_canvas
    if hasattr(_sdc, "_VALID_DRAWING_MODES"):
        _sdc._VALID_DRAWING_MODES = frozenset(_sdc._VALID_DRAWING_MODES | {"transform", "edit"})
    if hasattr(_sdc, "_REMOVED_DRAWING_MODES"):
        _sdc._REMOVED_DRAWING_MODES = {
            k: v for k, v in _sdc._REMOVED_DRAWING_MODES.items() if k not in ("transform", "edit")
        }
except Exception:
    st_canvas = None
    boxes_to_drawing = None

import importlib
import config
try:
    importlib.reload(config)
except Exception:
    pass

import persistence
try:
    importlib.reload(persistence)
except Exception:
    pass

import vision
try:
    importlib.reload(vision)
except Exception:
    pass

if "matcher" in st.session_state and st.session_state.matcher is not None:
    if not hasattr(st.session_state.matcher, "_proposals_opencv"):
        st.session_state.matcher = None

from config import (
    DEVICE_CHOICES,
    MODEL_CHOICES,
    MODEL_CLIP_REFERENCE,
    MODEL_GROUNDING_DINO,
    MODEL_INFO,
    PROPOSAL_METHOD_CHOICES,
    PROPOSAL_OPENCV,
    PROPOSAL_SAM,
    WORKSPACE_DIR,
    get_hardware_acceleration_tip,
    resolve_device,
)
from persistence import (
    WorkspaceError,
    WorkspaceValidationError,
    clear_saved_workspace,
    export_backup_zip,
    has_saved_workspace,
    load_workspace,
    restore_backup_zip,
    save_workspace,
)

from vision import (
    GroundingDinoDetector,
    ReferenceMatcher,
    crop_box,
    draw_box_with_label,
    evaluate_detections,
    export_yolo_zip,
    extract_class_list,
    generate_opencv_proposals,
    parse_canvas_objects,
    split_dataset,
    validate_dataset,
    yolo_line,
)

try:
    from vision import deduplicate_car_predictions, filter_one_car_per_image
except ImportError:
    def filter_one_car_per_image(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enforce strictly one most accurate car box per image."""
        car_preds = [p for p in predictions if str(p.get("class", "")).strip().lower() == "car"]
        if len(car_preds) <= 1:
            return predictions

        non_car_preds = [p for p in predictions if str(p.get("class", "")).strip().lower() != "car"]

        def _car_sort_key(p: Dict[str, Any]) -> Tuple[float, float]:
            box = p.get("box", [0, 0, 0, 0])
            w = max(0.0, float(box[2] - box[0])) if len(box) >= 4 else 0.0
            h = max(0.0, float(box[3] - box[1])) if len(box) >= 4 else 0.0
            area = w * h
            score = float(p.get("score", 0.0))
            return (score, area)

        best_car = max(car_preds, key=_car_sort_key)
        result = non_car_preds + [best_car]
        return sorted(result, key=lambda p: float(p.get("score", 0.0)), reverse=True)

    deduplicate_car_predictions = filter_one_car_per_image

# -----------------------------------------------------------------------------
# Streamlit Page Setup & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LabelForge · AI Auto-Labeling",
    page_icon="◩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');

.stApp {
    background-color: #f7f9f7;
    color: #17221d;
    font-family: 'DM Sans', sans-serif;
}

.hero {
    padding: 10px 0 20px;
    border-bottom: 1px solid #dde3de;
    margin-bottom: 22px;
}
.eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    color: #2b7a5a;
    text-transform: uppercase;
}
.hero h1 {
    font-size: 38px;
    font-weight: 800;
    margin: 6px 0 4px;
    color: #111a15;
}
.hero p {
    color: #5c6c63;
    font-size: 15px;
    margin: 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e1e7e2;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
.metric-card span {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    color: #728278;
    letter-spacing: 0.05em;
}
.metric-card strong {
    display: block;
    font-size: 26px;
    color: #111a15;
    margin-top: 4px;
}

.workflow-card {
    background: #ffffff;
    border: 1px solid #e1e7e2;
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
}
.workflow-step {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #2b7a5a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.workflow-title {
    font-size: 16px;
    font-weight: 700;
    color: #111a15;
    margin: 6px 0 8px;
}
.workflow-body {
    font-size: 13px;
    color: #5c6c63;
    line-height: 1.5;
}

.status-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 12px;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
}
.status-badge-green {
    background: #e2f4ea;
    color: #17653c;
    border: 1px solid #bce6cd;
}
.status-badge-blue {
    background: #e4effb;
    color: #1a518d;
    border: 1px solid #bfdbfe;
}
.status-badge-orange {
    background: #fef3e2;
    color: #92400e;
    border: 1px solid #fed7aa;
}

[data-testid="stSidebar"] {
    background-color: #eef2ed;
}
</style>
<div class="hero">
    <div class="eyebrow">Phase 4: Dataset Export, Validation, and Quality Checks · YOLO Compliant</div>
    <h1>LabelForge</h1>
    <p>Zero-shot text detection, iterative visual reference matching, pre-export validation, and reproducible YOLO dataset packaging.</p>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization & Local Workspace Restore
# -----------------------------------------------------------------------------
state_defaults = {
    "images": dict,
    "labels": lambda: defaultdict(list),
    "refs": lambda: defaultdict(list),
    "pending": lambda: defaultdict(list),
    "times": list,
    "rejections": list,
    "detector": lambda: None,
    "detector_device": lambda: None,
    "matcher": lambda: None,
    "matcher_device": lambda: None,
    "save_status": lambda: {"status": "Ready", "time": None, "error": None},
    "workspace_auto_restored": lambda: False,
    "persisted_settings": dict,
    "canvas_revision": dict,
    "open_ref_dialog": lambda: False,
    "ref_mgr_page": lambda: 1,
    "inline_ref_page": lambda: 1,
    "selected_image_override": lambda: None,
    "batch_folder_images": dict,
    "batch_folder_results": dict,
    "batch_inspect_image": lambda: None,
    "batch_last_folder_path": lambda: "",
}

for key, factory in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = factory()
st.session_state["open_ref_dialog"] = False

# Automatic restore: When the app starts or a browser session refreshes,
# restore the most recent workspace from the local save file without requiring re-upload.
if not st.session_state.workspace_auto_restored:
    st.session_state.workspace_auto_restored = True
    if not st.session_state.images and has_saved_workspace(WORKSPACE_DIR):
        try:
            restored = load_workspace(WORKSPACE_DIR)
            st.session_state.images = restored["images"]
            st.session_state.labels = restored["labels"]
            st.session_state.refs = restored["refs"]
            st.session_state.pending = restored["pending"]
            st.session_state.times = restored["times"]
            st.session_state.rejections = restored["rejections"]
            st.session_state.persisted_settings = restored.get("settings", {})
            st.session_state.save_status = {
                "status": "Saved",
                "time": restored.get("saved_at"),
                "error": None,
            }
        except Exception as exc:
            st.session_state.save_status = {
                "status": "Saving failed",
                "time": None,
                "error": f"Failed to restore workspace: {exc}",
            }


def open_folder_picker(initial_dir: str = "") -> str:
    """Open a native Windows folder picker popup dialog and return the selected path."""
    # 1. Try tkinter (built-in standard library, fast, native)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_p = initial_dir if (initial_dir and os.path.isdir(initial_dir)) else os.getcwd()
        path = filedialog.askdirectory(title="Select Folder Containing Images", initialdir=init_p)
        root.destroy()
        if path:
            return os.path.normpath(path)
    except Exception:
        pass

    # 2. Fallback to PowerShell FolderBrowserDialog
    try:
        import subprocess
        ps_cmd = (
            "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select Folder Containing Images'; "
            "$dialog.ShowNewFolderButton = $false; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dialog.SelectedPath }"
        )
        res = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, timeout=30)
        chosen = res.strip()
        if chosen and os.path.isdir(chosen):
            return os.path.normpath(chosen)
    except Exception:
        pass

    return ""


def persist_active_workspace(extra_settings: Dict[str, Any] | None = None) -> bool:
    """Save the active workspace to local disk using atomic writes.

    Guarantees no runtime ML models, PyTorch tensors, or CUDA objects are serialized.
    """
    try:
        active_settings = dict(st.session_state.get("persisted_settings", {}))
        if extra_settings:
            active_settings.update(extra_settings)

        res = save_workspace(
            workspace_dir=WORKSPACE_DIR,
            images=st.session_state.images,
            labels=st.session_state.labels,
            refs=st.session_state.refs,
            pending=st.session_state.pending,
            times=st.session_state.times,
            rejections=st.session_state.rejections,
            settings=active_settings,
        )
        st.session_state._backup_bytes_ready = None
        st.session_state.save_status = {
            "status": "Saved",
            "time": res["time"],
            "error": None,
        }
        return True
    except Exception as exc:
        st.session_state.save_status = {
            "status": "Saving failed",
            "time": st.session_state.save_status.get("time"),
            "error": str(exc),
        }
        return False


# -----------------------------------------------------------------------------
# Helper Functions for Class Name & YOLO ID Management
# -----------------------------------------------------------------------------
def get_known_classes() -> List[str]:
    """Gather all unique class names across labels, references, and pending predictions."""
    known: Set[str] = set()
    for rows in st.session_state.labels.values():
        for r in rows:
            if r.get("class"):
                known.add(r["class"])
    for cls in st.session_state.refs.keys():
        known.add(cls)
    for rows in st.session_state.pending.values():
        for r in rows:
            if r.get("class"):
                known.add(r["class"])
    return sorted(known)


def get_class_id_map() -> Dict[str, int]:
    """Map each unique class name to its zero-based integer YOLO class ID."""
    classes = get_known_classes()
    return {cls_name: i for i, cls_name in enumerate(classes)}


def get_all_reference_entries() -> List[Dict[str, Any]]:
    """Return all visual reference items with metadata and crop objects."""
    entries: List[Dict[str, Any]] = []
    for image_name, rows in st.session_state.labels.items():
        image = st.session_state.images.get(image_name)
        if image is None:
            continue
        for b_idx, row in enumerate(rows):
            box = row.get("box", ())
            cls_name = row.get("class", "")
            if cls_name and len(box) == 4:
                try:
                    box_id = row.get("id") or f"{image_name}_{b_idx}_{int(box[0])}_{int(box[1])}"
                    row["id"] = box_id
                    crop = row.get("_crop")
                    if crop is None:
                        crop = crop_box(image, box)
                        row["_crop"] = crop
                    entries.append({
                        "id": box_id,
                        "image_name": image_name,
                        "box_index": b_idx,
                        "box": list(box),
                        "class": str(cls_name),
                        "source": row.get("source", "manual"),
                        "score": row.get("score"),
                        "is_reference": bool(row.get("is_reference", True)),
                        "crop": crop,
                        "width": crop.width,
                        "height": crop.height,
                    })
                except Exception:
                    continue
    return entries


def delete_reference_entry(image_name: str, box_id: str, box: Sequence[float]) -> bool:
    """Permanently delete an approved annotation and its visual reference crop."""
    if image_name not in st.session_state.labels:
        return False
    rows = st.session_state.labels[image_name]
    new_rows = [
        r for r in rows
        if r.get("id") != box_id and list(r.get("box", [])) != list(box)
    ]
    if len(new_rows) < len(rows):
        st.session_state.labels[image_name] = new_rows
        st.session_state._reference_signature = None
        if st.session_state.get("matcher") is not None:
            st.session_state.matcher.invalidate_reference_cache()
        persist_active_workspace()
        return True
    return False


def toggle_reference_active(image_name: str, box_id: str, box: Sequence[float], is_ref: bool) -> bool:
    """Toggle whether an approved box is included in visual reference matching."""
    if image_name not in st.session_state.labels:
        return False
    for r in st.session_state.labels[image_name]:
        if r.get("id") == box_id or list(r.get("box", [])) == list(box):
            r["is_reference"] = is_ref
            st.session_state._reference_signature = None
            if st.session_state.get("matcher") is not None:
                st.session_state.matcher.invalidate_reference_cache()
            persist_active_workspace()
            return True
    return False


def delete_multiple_reference_entries(items: Sequence[Dict[str, Any]]) -> int:
    """Delete multiple reference entries in bulk with a single atomic disk save."""
    if not items:
        return 0
    items_to_del = {(item["image_name"], item.get("id")) for item in items}
    count = 0
    for img_name in {item["image_name"] for item in items}:
        if img_name not in st.session_state.labels:
            continue
        old_rows = st.session_state.labels[img_name]
        new_rows = [r for r in old_rows if (img_name, r.get("id")) not in items_to_del]
        count += len(old_rows) - len(new_rows)
        st.session_state.labels[img_name] = new_rows

    if count > 0:
        st.session_state._reference_signature = None
        if st.session_state.get("matcher") is not None:
            st.session_state.matcher.invalidate_reference_cache()
        persist_active_workspace()
    return count


def toggle_multiple_references(items: Sequence[Dict[str, Any]], is_ref: bool) -> int:
    """Toggle multiple references in bulk with a single atomic disk save."""
    if not items:
        return 0
    items_to_toggle = {(item["image_name"], item.get("id")) for item in items}
    count = 0
    for img_name in {item["image_name"] for item in items}:
        if img_name not in st.session_state.labels:
            continue
        for r in st.session_state.labels[img_name]:
            if (img_name, r.get("id")) in items_to_toggle:
                r["is_reference"] = is_ref
                count += 1

    if count > 0:
        st.session_state._reference_signature = None
        if st.session_state.get("matcher") is not None:
            st.session_state.matcher.invalidate_reference_cache()
        persist_active_workspace()
    return count


def _on_ref_dialog_dismiss() -> None:
    """Reset dialog open state on modal dismissal."""
    st.session_state.open_ref_dialog = False


@st.dialog("📚 Visual Reference Library Manager", width="large")
def reference_manager_dialog() -> None:
    """Interactive modal manager for visiting, inspecting, filtering, and deleting reference crops."""
    entries = get_all_reference_entries()
    if not entries:
        st.info("No visual reference crops stored yet in the workspace.")
        if st.button("Close Manager", key="dlg_empty_close", use_container_width=True):
            st.session_state.open_ref_dialog = False
            st.rerun()
        return

    total_refs = len(entries)
    active_refs = sum(1 for e in entries if e["is_reference"])
    excluded_refs = total_refs - active_refs
    unique_classes = sorted(list({e["class"] for e in entries}))

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Total Visual Crops", total_refs)
    col_m2.metric("Active in Matcher", active_refs)
    col_m3.metric("Excluded from Matcher", excluded_refs)
    col_m4.metric("Classes Represented", len(unique_classes))

    st.markdown("---")

    col_f1, col_f2, col_f3, col_f4 = st.columns([1.5, 1.5, 1.2, 1])
    with col_f1:
        cls_filter = st.selectbox(
            "Filter by Class",
            ["All Classes"] + unique_classes,
            key="ref_mgr_cls_filter",
        )
    with col_f2:
        img_filter = st.text_input(
            "Search Image Name",
            placeholder="e.g. car_001",
            key="ref_mgr_img_filter",
        ).strip().lower()
    with col_f3:
        status_filter = st.selectbox(
            "Filter Status",
            ["All", "Active Only", "Excluded Only"],
            key="ref_mgr_status_filter",
        )
    with col_f4:
        per_page = st.selectbox(
            "Per Page",
            [12, 24, 48],
            index=0,
            key="ref_mgr_per_page",
        )

    filtered = entries
    if cls_filter != "All Classes":
        filtered = [e for e in filtered if e["class"] == cls_filter]
    if img_filter:
        filtered = [e for e in filtered if img_filter in e["image_name"].lower()]
    if status_filter == "Active Only":
        filtered = [e for e in filtered if e["is_reference"]]
    elif status_filter == "Excluded Only":
        filtered = [e for e in filtered if not e["is_reference"]]

    with st.expander(f"⚡ Bulk Operations on Filtered ({len(filtered)} crops)", expanded=False):
        bcol1, bcol2, bcol3 = st.columns([1.6, 1.2, 1.2])
        with bcol1:
            confirm_del = st.checkbox(
                f"Confirm permanent deletion of all {len(filtered)} filtered crops",
                key="ref_mgr_confirm_bulk_del",
            )
            if st.button(
                f"🗑️ Delete All Filtered ({len(filtered)})",
                disabled=(not confirm_del or len(filtered) == 0),
                type="primary",
                key="btn_bulk_del_refs",
            ):
                num_del = delete_multiple_reference_entries(filtered)
                st.toast(f"Deleted {num_del} reference crops successfully!")
                st.rerun()
        with bcol2:
            if st.button(
                f"🚫 Exclude All ({len(filtered)})",
                disabled=(len(filtered) == 0),
                key="btn_bulk_exclude_refs",
                help="Stop using these crops as references without deleting their label annotations",
            ):
                num_toggled = toggle_multiple_references(filtered, False)
                st.toast(f"Excluded {num_toggled} crops from reference matching!")
                st.rerun()
        with bcol3:
            if st.button(
                f"✅ Include All ({len(filtered)})",
                disabled=(len(filtered) == 0),
                key="btn_bulk_include_refs",
                help="Re-enable these crops for visual reference matching",
            ):
                num_toggled = toggle_multiple_references(filtered, True)
                st.toast(f"Included {num_toggled} crops in reference matching!")
                st.rerun()

    if not filtered:
        st.warning("No reference crops match the selected filters.")
        if st.button("Close Manager", key="dlg_nofilter_close"):
            st.session_state.open_ref_dialog = False
            st.rerun()
        return

    total_pages = max(1, (len(filtered) + per_page - 1) // per_page)
    if "ref_mgr_page" not in st.session_state:
        st.session_state.ref_mgr_page = 1
    if st.session_state.ref_mgr_page > total_pages:
        st.session_state.ref_mgr_page = total_pages

    curr_page = st.session_state.ref_mgr_page
    start_idx = (curr_page - 1) * per_page
    end_idx = min(start_idx + per_page, len(filtered))
    page_items = filtered[start_idx:end_idx]

    num_cols = 3
    for row_start in range(0, len(page_items), num_cols):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            item_idx = row_start + col_idx
            if item_idx < len(page_items):
                item = page_items[item_idx]
                with cols[col_idx]:
                    with st.container(border=True):
                        h_left, h_right = st.columns([1.5, 1])
                        with h_left:
                            st.markdown(f"**🏷️ `{item['class']}`**")
                        with h_right:
                            if item["is_reference"]:
                                st.markdown("<span style='color:#17653c;font-weight:600;font-size:12px;'>🟢 Active</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span style='color:#92400e;font-weight:600;font-size:12px;'>⚪ Excluded</span>", unsafe_allow_html=True)

                        st.image(item["crop"], use_container_width=True)

                        score_info = f" · score: {item['score']:.2f}" if item.get("score") else ""
                        st.caption(
                            f"📷 **{item['image_name']}** · *{item['source']}{score_info}*\n\n"
                            f"📐 `{item['box']}` · `{item['width']}×{item['height']}px`"
                        )

                        btn_c1, btn_c2 = st.columns([1, 1])
                        with btn_c1:
                            if st.button("🗑️ Delete", key=f"del_ref_{item['id']}", help="Permanently delete this reference crop and annotation", use_container_width=True):
                                delete_reference_entry(item["image_name"], item["id"], item["box"])
                                st.toast(f"Deleted crop from {item['image_name']}!")
                                st.rerun()
                        with btn_c2:
                            if item["is_reference"]:
                                if st.button("🚫 Exclude", key=f"toggle_ref_{item['id']}", help="Keep label in dataset but exclude from visual matching", use_container_width=True):
                                    toggle_reference_active(item["image_name"], item["id"], item["box"], False)
                                    st.toast("Excluded from visual matching!")
                                    st.rerun()
                            else:
                                if st.button("✅ Include", key=f"toggle_ref_{item['id']}", help="Include as active visual reference for matcher", use_container_width=True):
                                    toggle_reference_active(item["image_name"], item["id"], item["box"], True)
                                    st.toast("Included in visual matching!")
                                    st.rerun()

                        if st.button("🎯 Select Image", key=f"jump_{item['id']}", help="Set this image as active in workspace", use_container_width=True):
                            st.session_state.selected_image_override = item["image_name"]
                            st.session_state.open_ref_dialog = False
                            st.rerun()

    st.markdown("---")
    p_prev, p_info, p_next = st.columns([1, 2, 1])
    with p_prev:
        if st.button("⬅️ Previous", disabled=(curr_page <= 1), key="ref_mgr_btn_prev", use_container_width=True):
            st.session_state.ref_mgr_page = max(1, curr_page - 1)
            st.rerun()
    with p_info:
        st.markdown(
            f"<div style='text-align:center;padding-top:6px;font-family:monospace;font-size:13px;'>"
            f"Page <b>{curr_page}</b> of <b>{total_pages}</b> · Showing <b>{start_idx + 1}–{end_idx}</b> of <b>{len(filtered)}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with p_next:
        if st.button("Next ➡️", disabled=(curr_page >= total_pages), key="ref_mgr_btn_next", use_container_width=True):
            st.session_state.ref_mgr_page = min(total_pages, curr_page + 1)
            st.rerun()

    if st.button("Done / Close Manager", key="ref_mgr_btn_close", use_container_width=True):
        st.session_state.open_ref_dialog = False
        st.rerun()


# -----------------------------------------------------------------------------
# Module-level defaults for Model & Hardware Settings
# -----------------------------------------------------------------------------
_persisted_s_init = st.session_state.get("persisted_settings", {})
_init_mc = _persisted_s_init.get("model_choice", MODEL_CLIP_REFERENCE)
if _init_mc in (
    "Visual reference matcher (iterative CLIP)",
    "Visual reference matcher (CLIP)",
    "Visual reference matcher (SAM + DINOv2)",
    "Visual reference matcher",
):
    _init_mc = MODEL_CLIP_REFERENCE
elif _init_mc in (
    "Grounding DINO Tiny (zero-shot text detector)",
    "Grounding DINO",
    "Grounding DINO Tiny",
):
    _init_mc = MODEL_GROUNDING_DINO
if _init_mc not in MODEL_CHOICES:
    _init_mc = MODEL_CLIP_REFERENCE if MODEL_CLIP_REFERENCE in MODEL_CHOICES else MODEL_CHOICES[0]
model_choice = _init_mc
device_choice = _persisted_s_init.get("device_choice", "CUDA")
resolved_device, device_status = resolve_device(device_choice)
info = MODEL_INFO.get(model_choice, {})
clip_threshold = float(_persisted_s_init.get("clip_threshold", 0.25))
detail_proposals = int(_persisted_s_init.get("detail_proposals", 240))
car_nms_iou = float(_persisted_s_init.get("car_nms_iou", 0.50))
proposal_method_setting = _persisted_s_init.get("proposal_method", PROPOSAL_OPENCV)
if proposal_method_setting not in PROPOSAL_METHOD_CHOICES:
    proposal_method_setting = PROPOSAL_OPENCV

# -----------------------------------------------------------------------------
# Sidebar Configuration & Image Upload
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Image Workspace")
    uploads = st.file_uploader(
        "Upload dataset images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Images stay in your local browser session and are processed locally.",
    )

    if uploads:
        new_uploaded = False
        for u in uploads:
            if u.name not in st.session_state.images:
                st.session_state.images[u.name] = Image.open(u).convert("RGB")
                st.session_state.labels[u.name] = []
                new_uploaded = True
        if new_uploaded:
            persist_active_workspace()

    image_names: List[str] = list(st.session_state.images.keys())

    # Derive the shared reference bank from approved labels.
    reference_signature = tuple(sorted(
        (image_name, str(row.get("class", "")), tuple(row.get("box", ())), bool(row.get("is_reference", True)))
        for image_name, rows in st.session_state.labels.items()
        for row in rows if row.get("class") and len(row.get("box", ())) == 4 and row.get("is_reference", True)
    ))
    if st.session_state.get("_reference_signature") != reference_signature:
        rebuilt_refs = defaultdict(list)
        for image_name, rows in st.session_state.labels.items():
            image = st.session_state.images.get(image_name)
            if image is None: continue
            for row in rows:
                box = row.get("box", ())
                if row.get("class") and len(box) == 4 and row.get("is_reference", True):
                    try: rebuilt_refs[str(row["class"])].append(crop_box(image, box))
                    except (TypeError, ValueError, IndexError): continue
        st.session_state.refs = rebuilt_refs
        st.session_state._reference_signature = reference_signature

    if image_names:
        default_img_idx = 0
        override_img = st.session_state.get("selected_image_override")
        if override_img and override_img in image_names:
            default_img_idx = image_names.index(override_img)
            st.session_state.selected_image_override = None

        selected_image_name = st.selectbox(
            "Active Image",
            image_names,
            index=default_img_idx,
            format_func=lambda n: f"{n} · {len(st.session_state.labels[n])} labels",
        )
    else:
        selected_image_name = None
        st.info("Upload images to begin your labeling workspace.")

    sidebar_class_map = get_class_id_map()
    if sidebar_class_map:
        with st.expander(f"🏷️ YOLO Class IDs ({len(sidebar_class_map)})", expanded=True):
            for cls_name, cid in sidebar_class_map.items():
                st.markdown(f"- `ID {cid}` : **{cls_name}**")

    # -------------------------------------------------------------------------
    # Sidebar: Local Workspace Persistence & Recovery Controls
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💾 Local Workspace & Backup")

    curr_status = st.session_state.save_status.get("status", "Ready")
    last_time = st.session_state.save_status.get("time")
    status_err = st.session_state.save_status.get("error")

    if curr_status == "Saved":
        st.markdown(
            f'<div style="padding: 8px 12px; border-radius: 8px; background: #e2f4ea; border: 1px solid #bce6cd; color: #17653c; font-size: 13px; font-weight: 600;">'
            f'🟢 Saved <span style="font-size: 11px; font-weight: 400; color: #2e7d32; display: block;">Last autosave: {last_time or "Just now"}</span></div>',
            unsafe_allow_html=True,
        )
    elif curr_status == "Saving failed":
        st.markdown(
            f'<div style="padding: 8px 12px; border-radius: 8px; background: #fde8e8; border: 1px solid #f8b4b4; color: #9b1c1c; font-size: 13px; font-weight: 600;">'
            f'🔴 Saving failed <span style="font-size: 11px; font-weight: 400; color: #c81e1e; display: block;">{status_err or "Unknown error"}</span></div>',
            unsafe_allow_html=True,
        )
    elif curr_status in ("Cleared", "Cleared from disk"):
        st.markdown(
            f'<div style="padding: 8px 12px; border-radius: 8px; background: #f3f4f6; border: 1px solid #e5e7eb; color: #4b5563; font-size: 13px; font-weight: 600;">'
            f'⚪ {curr_status}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Auto-saves locally on every change using atomic file writes.")

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("💾 Save Now", use_container_width=True, help="Force an immediate atomic autosave to disk"):
            if persist_active_workspace():
                st.toast("Workspace saved locally!")
                st.rerun()

    with c_btn2:
        if st.session_state.images:
            if st.session_state.get("_backup_bytes_ready") is not None:
                st.download_button(
                    "📥 Download (.zip)",
                    data=st.session_state._backup_bytes_ready,
                    file_name="labelforge-workspace-backup.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Click to download your prepared workspace backup ZIP",
                )
            else:
                if st.button("📦 Prepare Zip", use_container_width=True, help="Package full workspace into a downloadable ZIP"):
                    with st.spinner("Packaging workspace backup..."):
                        st.session_state._backup_bytes_ready = export_backup_zip(
                            images=st.session_state.images,
                            labels=st.session_state.labels,
                            refs=st.session_state.refs,
                            pending=st.session_state.pending,
                            times=st.session_state.times,
                            rejections=st.session_state.rejections,
                            settings=st.session_state.persisted_settings,
                        )
                    st.rerun()
        else:
            st.button("📥 Backup (.zip)", disabled=True, use_container_width=True, help="Upload images first to download a project backup")

    total_sidebar_refs = sum(len(crops) for crops in st.session_state.refs.values())
    if st.button(f"📚 Reference Library ({total_sidebar_refs})", use_container_width=True, help="Inspect, browse, and delete visual reference crops", key="sidebar_ref_lib_trigger"):
        reference_manager_dialog()

    with st.expander("🔄 Restore from Backup", expanded=False):
        st.caption("Restore a previous LabelForge project backup ZIP. Validation runs before restoring to prevent corruption.")
        backup_file = st.file_uploader(
            "Upload project backup archive (.zip)",
            type=["zip"],
            key="project_backup_uploader",
            help="Select a previously exported labelforge-workspace-backup.zip file.",
        )
        if backup_file is not None:
            if st.button("Apply Restored Backup", type="primary", use_container_width=True):
                try:
                    restored_proj = restore_backup_zip(backup_file.getvalue())
                    st.session_state.images = restored_proj["images"]
                    st.session_state.labels = restored_proj["labels"]
                    st.session_state.refs = restored_proj["refs"]
                    st.session_state.pending = restored_proj["pending"]
                    st.session_state.times = restored_proj["times"]
                    st.session_state.rejections = restored_proj["rejections"]
                    st.session_state.persisted_settings = restored_proj.get("settings", {})
                    # Immediately autosave to disk
                    persist_active_workspace()
                    st.success(f"Restored {len(restored_proj['images'])} images and annotations successfully!")
                    st.rerun()
                except WorkspaceValidationError as wve:
                    st.error(f"Cannot restore backup: {wve}")
                except Exception as exc:
                    st.error(f"Failed to read backup file: {exc}")

    with st.expander("🗑️ Clear Workspace", expanded=False):
        st.caption("Remove saved workspace data. Choose scope carefully.")
        clear_scope = st.radio(
            "Clear action",
            [
                "Clear disk autosave only (keeps current in-memory workspace)",
                "Reset entire workspace (clears disk AND resets in-memory session)",
            ],
            key="clear_workspace_scope",
        )
        confirm_clear = st.checkbox("Confirm clearing workspace", key="confirm_clear_ws_check")
        if st.button("Execute Clear", type="secondary", disabled=not confirm_clear, use_container_width=True):
            try:
                clear_saved_workspace(WORKSPACE_DIR)
                if "Reset entire workspace" in clear_scope:
                    st.session_state.images = {}
                    st.session_state.labels = defaultdict(list)
                    st.session_state.refs = defaultdict(list)
                    st.session_state.pending = defaultdict(list)
                    st.session_state.times = []
                    st.session_state.rejections = []
                    st.session_state.persisted_settings = {}
                    st.session_state.save_status = {"status": "Cleared", "time": None, "error": None}
                    st.success("Workspace reset completely!")
                else:
                    st.session_state.save_status = {"status": "Cleared from disk", "time": None, "error": None}
                    st.info("Cleared autosaved disk files. In-memory session was preserved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Clear failed: {exc}")

    st.markdown("---")
    st.markdown("### ⚙️ Model & Hardware Setup")

    persisted_s = st.session_state.persisted_settings
    saved_mc = persisted_s.get("model_choice")
    if saved_mc in (
        "Visual reference matcher (iterative CLIP)",
        "Visual reference matcher (CLIP)",
        "Visual reference matcher (SAM + DINOv2)",
        "Visual reference matcher",
    ):
        saved_mc = MODEL_CLIP_REFERENCE
    elif saved_mc in (
        "Grounding DINO Tiny (zero-shot text detector)",
        "Grounding DINO",
        "Grounding DINO Tiny",
    ):
        saved_mc = MODEL_GROUNDING_DINO
    if saved_mc not in MODEL_CHOICES:
        saved_mc = MODEL_CLIP_REFERENCE if MODEL_CLIP_REFERENCE in MODEL_CHOICES else MODEL_CHOICES[0]

    # Sanitize any previous widget session state so old names never appear in sidebar
    if "sb_model_mode_selector" in st.session_state:
        cur_sb = st.session_state["sb_model_mode_selector"]
        if cur_sb not in MODEL_CHOICES:
            if "visual" in str(cur_sb).lower():
                st.session_state["sb_model_mode_selector"] = MODEL_CLIP_REFERENCE
            else:
                st.session_state["sb_model_mode_selector"] = MODEL_GROUNDING_DINO

    def_model_idx = MODEL_CHOICES.index(saved_mc)
    model_choice = st.selectbox(
        "Automated Model Mode",
        MODEL_CHOICES,
        index=def_model_idx,
        help=(
            "Choose between Visual Reference Matcher (detect by matching against approved visual crops) "
            "or Text-Prompt Object Detector (detect by class name text prompts)."
        ),
        key="sb_model_mode_selector",
    )
    st.session_state.persisted_settings["model_choice"] = model_choice

    def_dev_idx = DEVICE_CHOICES.index(persisted_s["device_choice"]) if persisted_s.get("device_choice") in DEVICE_CHOICES else 0
    device_choice = st.selectbox(
        "Compute Device",
        DEVICE_CHOICES,
        index=def_dev_idx,
        help="Auto selects CUDA if a compatible GPU and PyTorch build are detected; otherwise falls back to CPU.",
    )
    st.session_state.persisted_settings["device_choice"] = device_choice

    resolved_device, device_status = resolve_device(device_choice)
    st.caption(f"**Device status:** {device_status}")

    # Display hardware acceleration advice if NVIDIA GPU exists on system but PyTorch is CPU-only
    gpu_tip = get_hardware_acceleration_tip()
    if gpu_tip:
        with st.container(border=True):
            st.markdown("⚡ **GPU Acceleration Available!**")
            st.caption(gpu_tip)

    info = MODEL_INFO.get(model_choice, {})
    if resolved_device is None:
        st.warning(
            "PyTorch is not installed in the active environment. "
            "Manual annotation remains active. To enable automated detection, install: "
            "`pip install -r requirements-model.txt`."
        )
    else:
        if model_choice == MODEL_GROUNDING_DINO:
            st.caption("✨ **Text-Prompt Detector:** Detects objects using natural language class prompts.")
        else:
            st.caption("✨ **Visual Reference Matcher:** Compares region proposals against your approved visual crops using cosine similarity.")

    st.caption("🔒 **Weight Guarantee:** Pretrained model weights are never retrained or modified during annotation.")

    st.markdown("---")
    st.markdown("### 🎛️ Detection Controls")

    if model_choice == MODEL_GROUNDING_DINO:
        dino_speed_preset = st.select_slider(
            "Inference Speed / Resolution",
            options=["Fast (640px · 2.5x faster)", "Balanced (800px)", "High Detail (1333px)"],
            value=persisted_s.get("dino_speed_preset", "Fast (640px · 2.5x faster)" if resolved_device == "cpu" else "Balanced (800px)"),
            help="Fast scales down the inference resolution to 640px. Recommended for CPU to dramatically accelerate inference with minimal accuracy impact.",
        )
        st.session_state.persisted_settings["dino_speed_preset"] = dino_speed_preset
        dino_target_size = 640 if "640" in dino_speed_preset else (800 if "800" in dino_speed_preset else None)

        dino_box_thresh = st.slider(
            "Box Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(persisted_s.get("dino_box_thresh", 0.35)),
            step=0.01,
            help="Confidence threshold for bounding box detection. Not a calibrated probability.",
        )
        dino_text_thresh = st.slider(
            "Text Alignment Threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(persisted_s.get("dino_text_thresh", 0.25)),
            step=0.01,
            help="Threshold for associating detected regions with input text phrases.",
        )
        st.session_state.persisted_settings["dino_box_thresh"] = dino_box_thresh
        st.session_state.persisted_settings["dino_text_thresh"] = dino_text_thresh
    else:
        proposal_method = st.radio(
            "Region Proposal Generator",
            options=PROPOSAL_METHOD_CHOICES,
            index=0 if persisted_s.get("proposal_method", PROPOSAL_OPENCV) == PROPOSAL_OPENCV else 1,
            help=(
                "OpenCV (Fast / Lightweight): Proposes object candidate regions using MSER, edge morphology, and multi-scale contours (~0.5s/image).\n"
                "Deep Visual Masks (High Detail): Proposes object regions using automatic neural masks (~5-10s/image)."
            ),
        )
        clip_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.10,
            max_value=0.95,
            value=float(persisted_s.get("clip_threshold", 0.45)),
            step=0.01,
            help="Calibrated similarity threshold with scene background rejection, geometric alignment, and multi-class competition.",
        )
        detail_proposals = st.select_slider(
            "Proposal Search Detail",
            options=[120, 240, 400, 600],
            value=int(persisted_s.get("detail_proposals", 240)) if persisted_s.get("detail_proposals") in [120, 240, 400, 600] else 240,
            help="Number of candidate sliding-window and selective-search regions evaluated per image.",
        )
        car_nms_iou = st.slider(
            "Car Overlap Deduplication (IoU)",
            min_value=0.10,
            max_value=0.90,
            value=float(persisted_s.get("car_nms_iou", 0.50)),
            step=0.05,
            help=(
                "IoU threshold for deduplicating overlapping car predictions. Among overlapping "
                "proposals matching the same car, the highest reference-similarity box is retained. "
                "Separate cars remain detectable."
            ),
        )
        st.session_state.persisted_settings["proposal_method"] = proposal_method
        st.session_state.persisted_settings["clip_threshold"] = clip_threshold
        st.session_state.persisted_settings["detail_proposals"] = detail_proposals
        st.session_state.persisted_settings["car_nms_iou"] = car_nms_iou

# -----------------------------------------------------------------------------
# Main Panel View
# -----------------------------------------------------------------------------
if not image_names:
    # Polished Onboarding & Foundation Workspace Screen
    st.markdown("### Welcome to LabelForge")
    st.markdown(
        "LabelForge is an AI-assisted annotation studio designed for high-efficiency bounding box creation. "
        "It leverages **frozen pretrained vision models** so you can auto-label datasets without GPU retraining or complex weight tuning."
    )

    # System Status Summary Bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            '<div class="metric-card"><span>Current Phase</span><strong>Phase 4</strong>'
            '<span class="status-badge status-badge-green">Export & Quality Active</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        dev_label = "GPU Accelerated" if resolved_device == "cuda" else "CPU Mode"
        badge_cls = "status-badge-green" if resolved_device == "cuda" else "status-badge-blue"
        st.markdown(
            f'<div class="metric-card"><span>Compute Device</span><strong>{dev_label}</strong>'
            f'<span class="status-badge {badge_cls}">{resolved_device or "No PyTorch"}</span></div>',
            unsafe_allow_html=True,
        )
    with c3:
        model_mode_short = "Text Prompts" if model_choice == MODEL_GROUNDING_DINO else "Visual Reference"
        st.markdown(
            f'<div class="metric-card"><span>Active Detector</span><strong>{model_mode_short}</strong>'
            f'<span class="status-badge status-badge-green">{info.get("status", "Active")}</span></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            '<div class="metric-card"><span>Model Weights</span><strong>Frozen</strong>'
            '<span class="status-badge status-badge-green">Zero-Shot / No Fine-Tuning</span></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### 🚀 Workflow at a Glance")

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        st.markdown(
            """<div class="workflow-card">
            <div class="workflow-step">Step 01</div>
            <div class="workflow-title">Upload Images</div>
            <div class="workflow-body">Load raw dataset images via the sidebar uploader. Supported formats: JPG, PNG, WEBP.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div class="workflow-card">
            <div class="workflow-step">Step 02</div>
            <div class="workflow-title">Prompt / Teach</div>
            <div class="workflow-body">Enter text prompts or draw seed bounding boxes (Visual Reference Matcher).</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """<div class="workflow-card">
            <div class="workflow-step">Step 03</div>
            <div class="workflow-title">Auto-Detect</div>
            <div class="workflow-body">Frozen vision models evaluate candidate regions and propose matching bounding boxes.</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_d:
        st.markdown(
            """<div class="workflow-card">
            <div class="workflow-step">Step 04</div>
            <div class="workflow-title">Iterative Review</div>
            <div class="workflow-body">Approve accurate boxes. Approved crops automatically expand the visual reference library!</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_e:
        st.markdown(
            """<div class="workflow-card">
            <div class="workflow-step">Step 05</div>
            <div class="workflow-title">Export YOLO</div>
            <div class="workflow-body">Download a complete, normalized YOLO dataset ZIP archive with labels and dataset.yaml.</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("ℹ️ Architecture & Design Principles", expanded=True):
        st.markdown(
            """
            - **Dual Automated Modes:** Detect objects via natural language text prompts or visual similarity matching using candidate region proposals + exemplar features.
            - **Iterative Feedback Loop:** Every approved bounding box (manual or automated) adds its crop to the visual reference library, improving future visual matching scans without altering model weights.
            - **Frozen Pretrained Models:** Model weights remain strictly frozen. The system operates purely in zero-shot inference mode.
            - **Lazy Resource Loading:** Checkpoints are never downloaded during app startup. Machine learning weights load strictly on-demand when detection is started.
            - **Hardware Flexibility & Low-Memory Optimization:** CUDA GPU acceleration supported with automatic memory cleanup and graceful CPU fallback for ~4 GB GPUs.
            - **Human-in-the-Loop Quality:** Machine suggestions are always presented as *pending review*. Only verified annotations make it to final export.
            """
        )

    if has_saved_workspace(WORKSPACE_DIR):
        with st.container(border=True):
            st.markdown("#### 💾 Saved Local Workspace Found")
            st.markdown("A previously autosaved workspace is stored on disk in `work/workspace/`. You can reload your images, approved labels, pending proposals, and visual references without re-uploading.")
            if st.button("🔄 Reload Saved Workspace", type="primary"):
                try:
                    restored = load_workspace(WORKSPACE_DIR)
                    st.session_state.images = restored["images"]
                    st.session_state.labels = restored["labels"]
                    st.session_state.refs = restored["refs"]
                    st.session_state.pending = restored["pending"]
                    st.session_state.times = restored["times"]
                    st.session_state.rejections = restored["rejections"]
                    st.session_state.persisted_settings = restored.get("settings", {})
                    st.session_state.save_status = {
                        "status": "Saved",
                        "time": restored.get("saved_at"),
                        "error": None,
                    }
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to reload saved workspace: {exc}")
    else:
        st.info("👈 **Next Step:** Upload your dataset images in the sidebar to open the active labeling workspace.")
        with st.container(border=True):
            st.markdown("#### 📂 Load Local Folder of Images")
            st.caption("Browse or specify any folder on your computer to import images directly into the workspace.")
            wb_c1, wb_c2 = st.columns([1.5, 2])
            with wb_c1:
                if st.button("🗂️ Select Folder (Pop-up Window)...", key="btn_wb_browse_popup", type="primary", use_container_width=True):
                    picked = open_folder_picker()
                    if picked and os.path.isdir(picked):
                        v_exts = {".jpg", ".jpeg", ".png", ".webp"}
                        for f in sorted(os.listdir(picked)):
                            if os.path.splitext(f)[1].lower() in v_exts:
                                fp = os.path.join(picked, f)
                                try:
                                    with Image.open(fp) as im:
                                        st.session_state.images[f] = im.convert("RGB")
                                except Exception:
                                    pass
                        if st.session_state.images:
                            persist_active_workspace()
                            st.success(f"Imported {len(st.session_state.images)} images from '{picked}'!")
                            st.rerun()
            with wb_c2:
                wb_col_in, wb_col_btn = st.columns([2, 1])
                with wb_col_in:
                    wb_path = st.text_input("...or paste folder path:", placeholder=r"e.g. C:\Users\sarim\Pictures\cars", key="wb_folder_input")
                with wb_col_btn:
                    st.write("")
                    st.write("")
                    if st.button("📂 Import Path", key="btn_wb_load_folder", use_container_width=True):
                        clean_wb = wb_path.strip().strip('"').strip("'")
                        if clean_wb and os.path.isdir(clean_wb):
                            v_exts = {".jpg", ".jpeg", ".png", ".webp"}
                            for f in sorted(os.listdir(clean_wb)):
                                if os.path.splitext(f)[1].lower() in v_exts:
                                    fp = os.path.join(clean_wb, f)
                                    try:
                                        with Image.open(fp) as im:
                                            st.session_state.images[f] = im.convert("RGB")
                                    except Exception:
                                        pass
                            if st.session_state.images:
                                persist_active_workspace()
                                st.success(f"Imported {len(st.session_state.images)} images from folder!")
                                st.rerun()
                        else:
                            st.error(f"Folder not found: '{clean_wb}'")

else:
    # Active Workspace Screen (Images uploaded)
    # Enforce strictly one most accurate box per image across pending predictions
    for _img_name, _cands in list(st.session_state.pending.items()):
        if len(_cands) > 1:
            st.session_state.pending[_img_name] = filter_one_car_per_image(_cands)

    total_imgs = len(image_names)
    labeled_count = sum(bool(st.session_state.labels[n]) for n in image_names)
    pending_count = sum(bool(st.session_state.pending[n]) for n in image_names)
    # An image is remaining unlabeled only if it has NO approved labels AND NO pending predictions
    unlabeled_count = sum(1 for n in image_names if not st.session_state.labels[n] and not st.session_state.pending[n])

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(
        f'<div class="metric-card"><span>Labeled Images</span><strong>{labeled_count}</strong></div>',
        unsafe_allow_html=True,
    )
    mc2.markdown(
        f'<div class="metric-card"><span>Pending Review</span><strong>{pending_count}</strong></div>',
        unsafe_allow_html=True,
    )
    mc3.markdown(
        f'<div class="metric-card"><span>Remaining Unlabeled</span><strong>{unlabeled_count}</strong></div>',
        unsafe_allow_html=True,
    )
    mc4.markdown(
        f'<div class="metric-card"><span>Total Dataset</span><strong>{total_imgs}</strong></div>',
        unsafe_allow_html=True,
    )

    tab_teach, tab_review, tab_batch, tab_export = st.tabs(
        [
            "01 · Label & Teach",
            "02 · Review Predictions",
            "03 · Batch Folder Predict",
            "04 · Export & Metrics",
        ]
    )

    # -------------------------------------------------------------------------
    # Tab 1: Label & Teach
    # -------------------------------------------------------------------------
    with tab_teach:
        left_col, right_col = st.columns([1.2, 1])
        active_img = st.session_state.images[selected_image_name]

        with left_col:
            st.markdown("#### Manual Reference / Seed Bounding Box")
            st.caption("Draw tightly around an object to save a visual reference or manual label.")

            annotated_preview = active_img.copy()
            draw_ctx = ImageDraw.Draw(annotated_preview)
            class_map = get_class_id_map()
            for item in st.session_state.labels[selected_image_name]:
                cls_name = item.get("class", "object")
                cid = class_map.get(cls_name, "?")
                draw_box_with_label(
                    draw_ctx,
                    item["box"],
                    label_text=f"[ID {cid}] {cls_name}",
                    color="#2b7a5a",
                    line_width=max(2, annotated_preview.width // 350),
                )

            scale_factor = min(1.0, 680.0 / annotated_preview.width)
            display_size = (int(annotated_preview.width * scale_factor), int(annotated_preview.height * scale_factor))
            display_img = annotated_preview.resize(display_size)

            if st_canvas is None:
                st.error(
                    "Interactive drawing canvas requires `streamlit-drawable-canvas`. "
                    "Install app dependencies: `pip install -r requirements.txt`"
                )
                canvas_result = None
            else:
                canvas_result = st_canvas(
                    background_image=display_img,
                    drawing_mode="rect",
                    stroke_color="#2b7a5a",
                    stroke_width=3,
                    fill_color="rgba(43, 122, 90, 0.15)",
                    height=display_img.height,
                    width=display_img.width,
                    key=f"draw-{selected_image_name}-{len(st.session_state.labels[selected_image_name])}",
                )

            class_input = st.text_input(
                "Class Label",
                placeholder="e.g. bottle, car, cup",
                key=f"class-{selected_image_name}",
            ).strip()

            save_disabled = not class_input or canvas_result is None or canvas_result.json_data is None
            if st.button("Save Manual Box as Visual Reference", type="secondary", disabled=save_disabled):
                canvas_objects = canvas_result.json_data.get("objects", [])
                if canvas_objects:
                    last_obj = canvas_objects[-1]
                    scale_x = active_img.width / display_img.width
                    scale_y = active_img.height / display_img.height

                    x1 = max(0, int(last_obj["left"] * scale_x))
                    y1 = max(0, int(last_obj["top"] * scale_y))
                    x2 = min(active_img.width, int((last_obj["left"] + last_obj["width"]) * scale_x))
                    y2 = min(active_img.height, int((last_obj["top"] + last_obj["height"]) * scale_y))

                    if x2 > x1 and y2 > y1:
                        box_coords = [x1, y1, x2, y2]
                        st.session_state.labels[selected_image_name].append(
                            {"class": class_input, "box": box_coords, "source": "manual"}
                        )
                        st.session_state.refs[class_input].append(crop_box(active_img, box_coords))
                        st.session_state.matcher = None
                        persist_active_workspace()
                        st.success(f"Saved manual box and visual reference for '{class_input}'!")
                        st.rerun()

            # Active image annotation list displaying YOLO class IDs
            active_labels = st.session_state.labels[selected_image_name]
            if active_labels:
                st.markdown(f"##### 🏷️ Active Image Labels ({len(active_labels)})")
                for b_idx, item in enumerate(active_labels):
                    cls_name = item.get("class", "object")
                    cid = class_map.get(cls_name, "?")
                    col_t, col_c, col_d = st.columns([2, 3, 1])
                    col_t.markdown(f"**`Class ID: {cid}`** · **{cls_name}**")
                    col_c.caption(f"{item['box']} · *{item.get('source', 'manual')}*")
                    if col_d.button("✕", key=f"del_box_{selected_image_name}_{b_idx}", help=f"Delete box #{b_idx+1}"):
                        st.session_state.labels[selected_image_name].pop(b_idx)
                        persist_active_workspace()
                        st.rerun()

        with right_col:
            known_classes = get_known_classes()
            total_refs_count = sum(len(crops) for crops in st.session_state.refs.values())

            # -----------------------------------------------------------------
            # Visual Reference Gallery & Management (Phase 3)
            # -----------------------------------------------------------------
            st.markdown("#### 📚 Visual Reference Library")
            all_existing_refs = get_all_reference_entries()
            total_crops_count = len(all_existing_refs)

            if total_crops_count == 0:
                st.info(
                    "No visual reference crops stored yet. "
                    "Draw a box on the left, or approve an automated detection to start building your visual library."
                )
            else:
                active_ref_classes = [c for c, crops in st.session_state.refs.items() if crops]
                st.caption(
                    f"**{total_refs_count} active reference(s)** across **{len(active_ref_classes)} class(es)**. "
                    "Approved bounding boxes serve as visual reference exemplars for visual matching."
                )

                # Primary Action Button to visit and manage reference crops in full dialog
                if st.button(
                    f"🖼️ Visit & Delete Reference Crops ({total_crops_count})",
                    type="primary",
                    use_container_width=True,
                    help="Browse, inspect, filter, exclude, or delete visual reference crops in a dedicated manager",
                    key="btn_open_ref_manager_dialog",
                ):
                    reference_manager_dialog()

                # Inline expandable browser
                with st.expander("🔍 Quick Browser · Inspect & Delete Crops Inline", expanded=True):
                    # Filter by class if multiple
                    if len(active_ref_classes) > 1:
                        sel_inline_cls = st.selectbox("Inspect class", active_ref_classes, key="inline_ref_cls_sel")
                    elif active_ref_classes:
                        sel_inline_cls = active_ref_classes[0]
                    else:
                        sel_inline_cls = None

                    if sel_inline_cls:
                        class_entries = [e for e in all_existing_refs if e["class"] == sel_inline_cls]
                        inline_per_page = 6
                        inline_total_pages = max(1, (len(class_entries) + inline_per_page - 1) // inline_per_page)
                        if "inline_ref_page" not in st.session_state:
                            st.session_state.inline_ref_page = 1
                        if st.session_state.inline_ref_page > inline_total_pages:
                            st.session_state.inline_ref_page = inline_total_pages

                        in_page = st.session_state.inline_ref_page
                        in_start = (in_page - 1) * inline_per_page
                        in_end = min(in_start + inline_per_page, len(class_entries))
                        in_items = class_entries[in_start:in_end]

                        st.markdown(
                            f"**`Class: {sel_inline_cls}`** (`{len(class_entries)}` total crops · Page {in_page}/{inline_total_pages})"
                        )

                        # Display 3 columns of cards
                        in_cols_num = 3
                        for r_idx in range(0, len(in_items), in_cols_num):
                            c_cols = st.columns(in_cols_num)
                            for c_i in range(in_cols_num):
                                idx_item = r_idx + c_i
                                if idx_item < len(in_items):
                                    itm = in_items[idx_item]
                                    with c_cols[c_i]:
                                        with st.container(border=True):
                                            st.image(itm["crop"], use_container_width=True)
                                            st.caption(f"📷 `{itm['image_name']}`\n\n`{itm['width']}×{itm['height']}px`")
                                            if st.button("🗑️ Delete", key=f"inline_del_{itm['id']}", use_container_width=True, help="Delete this reference crop and annotation"):
                                                delete_reference_entry(itm["image_name"], itm["id"], itm["box"])
                                                st.toast(f"Deleted reference crop from {itm['image_name']}!")
                                                st.rerun()

                        # Inline page navigation
                        if inline_total_pages > 1:
                            in_c1, in_c2, in_c3 = st.columns([1, 2, 1])
                            with in_c1:
                                if st.button("◀ Prev", disabled=(in_page <= 1), key="inline_ref_prev", use_container_width=True):
                                    st.session_state.inline_ref_page = max(1, in_page - 1)
                                    st.rerun()
                            with in_c2:
                                st.caption(f"Page {in_page} of {inline_total_pages}")
                            with in_c3:
                                if st.button("Next ▶", disabled=(in_page >= inline_total_pages), key="inline_ref_next", use_container_width=True):
                                    st.session_state.inline_ref_page = min(inline_total_pages, in_page + 1)
                                    st.rerun()

            st.markdown("---")

            # -----------------------------------------------------------------
            # Automated Detection: Text-Prompt Mode
            # -----------------------------------------------------------------
            if model_choice == MODEL_GROUNDING_DINO:
                st.markdown("#### 🤖 Text-Prompt Object Detector")
                st.caption(
                    "Detects objects using open-vocabulary natural language text prompts. "
                    "Model weights remain strictly frozen; no training required."
                )

                default_prompt_val = ", ".join(known_classes) if known_classes else ""
                text_prompt_input = st.text_input(
                    "Target Class Names / Text Prompts",
                    value=default_prompt_val,
                    placeholder="e.g. pillow and bed, or bottle, cup, person",
                    help="Enter any number of class names separated by commas or 'and' (e.g. 'pillow and bed'). Each will be detected as a separate class.",
                ).strip()

                parsed_prompt_classes = extract_class_list(text_prompt_input)
                if parsed_prompt_classes:
                    st.caption(f"🎯 **Identified classes ({len(parsed_prompt_classes)}):** {', '.join(f'`{c}`' for c in parsed_prompt_classes)}")
                elif known_classes:
                    st.caption(f"💡 Detected classes from dataset: **{', '.join(known_classes)}**")

                scan_scope = st.radio(
                    "Images to scan",
                    ["Selected image only", "All unlabeled images"],
                    horizontal=True,
                    key="dino_scope",
                )

                dino_ready = bool(text_prompt_input) and (resolved_device is not None)

                if resolved_device is None:
                    st.warning("Install model dependencies to enable detection: `pip install -r requirements-model.txt`")

                if st.button(
                    "⚡ Detect Objects with Text Prompts",
                    type="primary",
                    disabled=not dino_ready,
                    use_container_width=True,
                ):
                    target_names = (
                        [selected_image_name]
                        if scan_scope == "Selected image only"
                        else [n for n in image_names if not st.session_state.labels[n] and not st.session_state.pending[n]]
                    )

                    if not target_names:
                        st.info("No unlabeled images available to scan. Select an image individually or review pending predictions.")
                    else:
                        try:
                            if (
                                st.session_state.get("detector") is None
                                or st.session_state.get("detector_device") != device_choice
                            ):
                                with st.spinner("Initializing Text-Prompt Detector (loading weights)..."):
                                    st.session_state.detector = GroundingDinoDetector(device=device_choice)
                                    st.session_state.detector_device = device_choice

                            detector: GroundingDinoDetector = st.session_state.detector

                            start_time = time.perf_counter()
                            found_total = 0
                            progress_bar = st.progress(0.0)
                            status_placeholder = st.empty()

                            for idx, img_name in enumerate(target_names):
                                status_placeholder.markdown(
                                    f"⏳ Processing **{img_name}** ({idx + 1}/{len(target_names)}) · "
                                    f"Found **{found_total}** candidate(s) so far..."
                                )
                                img_obj = st.session_state.images[img_name]
                                predictions = detector.predict(
                                    image=img_obj,
                                    prompt=text_prompt_input,
                                    box_threshold=dino_box_thresh,
                                    text_threshold=dino_text_thresh,
                                    target_size=dino_target_size,
                                )
                                # Assign stable unique candidate IDs for review widget state stability
                                for p_idx, p in enumerate(predictions):
                                    p["id"] = f"{img_name}_{p_idx}_{time.time()}"
                                st.session_state.pending[img_name] = predictions
                                found_total += len(predictions)
                                progress_bar.progress((idx + 1) / len(target_names))

                            duration = time.perf_counter() - start_time
                            status_placeholder.empty()
                            progress_bar.empty()
                            st.session_state.times.append({"seconds": duration, "images": len(target_names)})
                            persist_active_workspace()

                            st.success(
                                f"Discovered {found_total} candidate object(s) across {len(target_names)} image(s) in {duration:.1f}s! "
                                "Switch to the **'02 · Review Predictions'** tab to inspect and approve."
                            )
                        except Exception as exc:
                            st.session_state.detector = None
                            err_str = str(exc)
                            if "out of memory" in err_str.lower():
                                st.error(
                                    "🚨 **CUDA Out of Memory:** The GPU ran out of VRAM. "
                                    "Switch 'Compute Device' to **CPU** in the sidebar to run on system RAM, or resize input images."
                                )
                            else:
                                st.error(f"Detection error: {exc}")

                st.caption(
                    "Scores are text-image matching similarity metrics, not calibrated probabilities. "
                    "All proposals require human review."
                )

            # -----------------------------------------------------------------
            # Automated Detection: Visual Reference Matcher Mode (Phase 3)
            # -----------------------------------------------------------------
            else:
                st.markdown("#### 🔄 Visual Reference Matcher")
                active_prop_method = st.session_state.persisted_settings.get("proposal_method", PROPOSAL_OPENCV)
                st.caption(
                    f"Proposals: **{active_prop_method}** · Matching: **Frozen Visual Features** (Cosine Similarity). "
                    "Approved predictions continuously expand the visual reference bank."
                )
                st.info(
                    "🚗 **Single Most Accurate Box Mode:** Configured to return strictly the single most accurate car box "
                    "per image (highest reference-similarity score, with larger area as tie-breaker)."
                )
                if resolved_device == "cuda":
                    st.success(f"⚡ **Compute Acceleration:** CUDA Active ({device_status})")
                else:
                    st.info(f"ℹ️ **Compute Device:** {device_status}")

                if total_refs_count == 0:
                    st.warning(
                        "⚠️ **No visual references available.**\n"
                        "To use visual reference matching, teach the system by either:\n"
                        "1. Drawing a manual reference box on the left, or\n"
                        "2. Detecting objects with text prompts and approving them in Tab 02."
                    )
                else:
                    active_ref_list = sorted([c for c, ex in st.session_state.refs.items() if ex])
                    st.write(f"🎯 **Target classes ({len(active_ref_list)}):** {', '.join(active_ref_list)}")

                scan_scope = st.radio(
                    "Images to scan",
                    ["Selected image only", "All unlabeled images"],
                    horizontal=True,
                    key="clip_scope",
                )

                clip_ready = (total_refs_count > 0) and (resolved_device is not None)

                if resolved_device is None:
                    st.warning("Install model dependencies to enable visual matching: `pip install -r requirements-model.txt`")

                if st.button(
                    "🔄 Scan Images with Visual References",
                    type="primary",
                    disabled=not clip_ready,
                    use_container_width=True,
                ):
                    target_names = (
                        [selected_image_name]
                        if scan_scope == "Selected image only"
                        else [n for n in image_names if not st.session_state.labels[n] and not st.session_state.pending[n]]
                    )

                    if not target_names:
                        st.info("No unlabeled images available to scan. Select an image individually or review pending predictions.")
                    else:
                        try:
                            if (
                                st.session_state.get("matcher") is None
                                or st.session_state.get("matcher_device") != device_choice
                                or not hasattr(st.session_state.get("matcher"), "_proposals_opencv")
                            ):
                                with st.spinner("Initializing Visual Reference Matcher (loading vision models on first run)..."):
                                    st.session_state.matcher = ReferenceMatcher(device_choice)
                                    st.session_state.matcher_device = device_choice

                            matcher: ReferenceMatcher = st.session_state.matcher

                            start_time = time.perf_counter()
                            found_total = 0
                            progress_bar = st.progress(0.0)
                            status_placeholder = st.empty()

                            for idx, img_name in enumerate(target_names):
                                status_placeholder.markdown(
                                    f"⏳ Matching references in **{img_name}** ({idx + 1}/{len(target_names)}) · "
                                    f"Found **{found_total}** candidate(s) so far..."
                                )
                                chosen_prop_method = (
                                    "sam"
                                    if st.session_state.persisted_settings.get("proposal_method") == PROPOSAL_SAM
                                    else "opencv"
                                )
                                predict_kwargs = dict(
                                    threshold=clip_threshold,
                                    limit=detail_proposals,
                                    margin=0.05,
                                    car_nms_iou=car_nms_iou,
                                )
                                if "proposal_method" in inspect.signature(matcher.predict).parameters:
                                    predict_kwargs["proposal_method"] = chosen_prop_method
                                predictions = matcher.predict(
                                    st.session_state.images[img_name],
                                    st.session_state.refs,
                                    **predict_kwargs,
                                )
                                predictions = filter_one_car_per_image(predictions)
                                for p_idx, p in enumerate(predictions):
                                    p["id"] = f"{img_name}_{p_idx}_{time.time()}"
                                st.session_state.pending[img_name] = predictions
                                found_total += len(predictions)
                                progress_bar.progress((idx + 1) / len(target_names))

                            duration = time.perf_counter() - start_time
                            status_placeholder.empty()
                            progress_bar.empty()
                            st.session_state.times.append({"seconds": duration, "images": len(target_names)})
                            persist_active_workspace()

                            st.success(
                                f"Discovered {found_total} candidate(s) across {len(target_names)} image(s) in {duration:.1f}s! "
                                "Switch to **'02 · Review Predictions'** to approve or refine."
                            )
                        except Exception as exc:
                            st.session_state.matcher = None
                            err_str = str(exc)
                            if "out of memory" in err_str.lower():
                                st.error(
                                    "🚨 **CUDA Out of Memory:** The GPU ran out of VRAM. "
                                    "Switch 'Compute Device' to **CPU** in the sidebar, or lower proposal detail."
                                )
                            else:
                                st.error(f"Visual matching error: {exc}")

                st.caption("🔒 Model weights stay strictly frozen. Zero training or weight alteration takes place.")

    # -------------------------------------------------------------------------
    # Tab 2: Review Predictions
    # -------------------------------------------------------------------------
    with tab_review:
        pending_image_names = [n for n in image_names if st.session_state.pending[n]]
        total_pending_proposals = sum(len(st.session_state.pending[n]) for n in image_names)

        # ---------------------------------------------------------------------
        # Global Bulk Actions & Filter Bar
        # ---------------------------------------------------------------------
        with st.container(border=True):
            top_c1, top_c2, top_c3, top_c4 = st.columns([2.5, 1.2, 1.2, 1.5])
            with top_c1:
                st.markdown(
                    f"##### 📋 Review Predictions ({total_pending_proposals} pending across dataset)\n"
                    f"**{len(pending_image_names)}** image(s) have candidate predictions awaiting review."
                )
            with top_c2:
                if st.button(
                    f"✅ Accept All ({total_pending_proposals})",
                    type="primary",
                    disabled=(total_pending_proposals == 0),
                    use_container_width=True,
                    help="Accept and approve ALL pending predictions across all images in the dataset into approved labels",
                    key="global_accept_all_proposals",
                ):
                    accepted_count = 0
                    for n in image_names:
                        cands = list(st.session_state.pending[n])
                        if cands:
                            img_obj = st.session_state.images[n]
                            for c in cands:
                                app_entry = {**c, "source": "auto"}
                                st.session_state.labels[n].append(app_entry)
                                st.session_state.refs[c["class"]].append(crop_box(img_obj, c["box"]))
                                accepted_count += 1
                            st.session_state.pending[n] = []
                    # Invalidate embedding cache so next scan re-encodes the new references.
                    # Keep the matcher object alive to avoid re-downloading model weights.
                    if st.session_state.get("matcher") is not None:
                        st.session_state.matcher.invalidate_reference_cache()
                    persist_active_workspace()
                    st.success(f"🎉 Accepted and approved all {accepted_count} candidate proposal(s) into your dataset!")
                    st.rerun()

            with top_c3:
                if st.button(
                    f"🗑️ Delete All ({total_pending_proposals})",
                    type="secondary",
                    disabled=(total_pending_proposals == 0),
                    use_container_width=True,
                    help="Reject and delete ALL pending predictions across all images in the dataset",
                    key="global_delete_all_proposals",
                ):
                    del_count = total_pending_proposals
                    for n in image_names:
                        st.session_state.rejections.extend([1] * len(st.session_state.pending[n]))
                        st.session_state.pending[n] = []
                    # Rejected candidates do not become references; cache stays valid.
                    persist_active_workspace()
                    st.info(f"🗑️ Deleted and rejected all {del_count} candidate proposal(s) across the dataset.")
                    st.rerun()

            with top_c4:
                filter_pending_only = st.toggle(
                    "Filter to pending only",
                    value=bool(pending_image_names),
                    help="Toggle to show only images with pending candidate proposals, or all uploaded images.",
                    key="filter_pending_toggle",
                )

        selectable_images = pending_image_names if (filter_pending_only and pending_image_names) else image_names

        if not selectable_images:
            st.info("No candidates awaiting review. Generate predictions from '01 · Label & Teach', or uncheck 'Filter to pending only' to inspect and edit any image.")
        else:
            # Store selectable_images in session_state for on_click callbacks
            st.session_state._review_selectable_images = selectable_images

            if "review_image_widget" not in st.session_state or st.session_state.review_image_widget not in selectable_images:
                st.session_state.review_image_widget = selected_image_name if selected_image_name in selectable_images else selectable_images[0]

            def on_prev_click():
                sel_list = st.session_state.get("_review_selectable_images", [])
                if not sel_list:
                    return
                curr = st.session_state.get("review_image_widget", sel_list[0])
                curr_i = sel_list.index(curr) if curr in sel_list else 0
                prev_i = (curr_i - 1) % len(sel_list)
                st.session_state.review_image_widget = sel_list[prev_i]

            def on_next_click():
                sel_list = st.session_state.get("_review_selectable_images", [])
                if not sel_list:
                    return
                curr = st.session_state.get("review_image_widget", sel_list[0])
                curr_i = sel_list.index(curr) if curr in sel_list else 0
                next_i = (curr_i + 1) % len(sel_list)
                st.session_state.review_image_widget = sel_list[next_i]

            def on_accept_this_click():
                sel_list = st.session_state.get("_review_selectable_images", [])
                curr_name = st.session_state.get("review_image_widget")
                if not curr_name or curr_name not in st.session_state.images:
                    return
                cands = list(st.session_state.pending.get(curr_name, []))
                if cands:
                    img_obj = st.session_state.images[curr_name]
                    for c in cands:
                        app_entry = {**c, "source": "auto"}
                        st.session_state.labels[curr_name].append(app_entry)
                        st.session_state.refs[c["class"]].append(crop_box(img_obj, c["box"]))
                    st.session_state.pending[curr_name] = []
                    if "canvas_revision" in st.session_state:
                        st.session_state.canvas_revision[curr_name] = st.session_state.canvas_revision.get(curr_name, 0) + 1
                    st.session_state.matcher = None
                    persist_active_workspace()
                rem_pending = [n for n in sel_list if st.session_state.pending.get(n)]
                if rem_pending:
                    st.session_state.review_image_widget = rem_pending[0]

            def on_delete_this_click():
                sel_list = st.session_state.get("_review_selectable_images", [])
                curr_name = st.session_state.get("review_image_widget")
                if not curr_name:
                    return
                cands = list(st.session_state.pending.get(curr_name, []))
                if cands:
                    st.session_state.rejections.extend([1] * len(cands))
                    st.session_state.pending[curr_name] = []
                    if "canvas_revision" in st.session_state:
                        st.session_state.canvas_revision[curr_name] = st.session_state.canvas_revision.get(curr_name, 0) + 1
                    st.session_state.matcher = None
                    persist_active_workspace()
                rem_pending = [n for n in sel_list if st.session_state.pending.get(n)]
                if rem_pending:
                    st.session_state.review_image_widget = rem_pending[0]

            curr_idx = selectable_images.index(st.session_state.review_image_widget)
            review_name = st.session_state.review_image_widget
            review_candidates = st.session_state.pending[review_name]
            cand_count_this = len(review_candidates)

            # -----------------------------------------------------------------
            # Navigation Toolbar with Arrow Keys (Left & Right shortcuts) + Active Image Bulk Actions
            # -----------------------------------------------------------------
            nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([1.1, 3.2, 1.1, 1.5, 1.5])
            with nav_c1:
                st.button(
                    "⬅️ Prev",
                    shortcut="Left",
                    on_click=on_prev_click,
                    use_container_width=True,
                    help="Previous image (Shortcut: Left Arrow ←)",
                    key="nav_btn_prev",
                )

            with nav_c2:
                st.selectbox(
                    f"Review Image ({curr_idx + 1} of {len(selectable_images)})",
                    selectable_images,
                    index=curr_idx,
                    key="review_image_widget",
                    format_func=lambda n: f"[{selectable_images.index(n) + 1}/{len(selectable_images)}] {n} · {len(st.session_state.pending[n])} pending · {len(st.session_state.labels[n])} approved",
                    label_visibility="collapsed",
                )

            with nav_c3:
                st.button(
                    "Next ➡️",
                    shortcut="Right",
                    on_click=on_next_click,
                    use_container_width=True,
                    help="Next image (Shortcut: Right Arrow →)",
                    key="nav_btn_next",
                )

            with nav_c4:
                st.button(
                    f"✅ Accept Image ({cand_count_this})",
                    type="primary",
                    disabled=(cand_count_this == 0),
                    on_click=on_accept_this_click,
                    use_container_width=True,
                    help=f"Accept all {cand_count_this} proposal(s) on this active image",
                    key=f"nav_accept_this_{review_name}",
                )

            with nav_c5:
                st.button(
                    f"🗑️ Delete Image ({cand_count_this})",
                    type="secondary",
                    disabled=(cand_count_this == 0),
                    on_click=on_delete_this_click,
                    use_container_width=True,
                    help=f"Delete all {cand_count_this} proposal(s) on this active image",
                    key=f"nav_delete_this_{review_name}",
                )

            st.caption("💡 **Navigation & Shortcuts:** Use **⬅️ Prev** / **Next ➡️** buttons or press **Left Arrow (←)** / **Right Arrow (→)** to quickly flip through images.")

            review_img = st.session_state.images[review_name]
            review_candidates = st.session_state.pending[review_name]
            approved_boxes = st.session_state.labels[review_name]
            class_map = get_class_id_map()
            base_known_classes = set(get_known_classes())

            # -----------------------------------------------------------------
            # Image Display & Interactive Canvas (Direct On-Image Edit, Move, Resize, Draw)
            # -----------------------------------------------------------------
            st.markdown(f"#### 🖼️ Image Inspection & Interactive Canvas (`{review_name}`)")
            st.caption(
                "**Legend:** 🟢 Green boxes = Approved labels. 🟠 Orange boxes = Pending proposals. "
                "💡 **Direct Editing:** In Edit mode, click any box directly on the image to drag its position or pull its handles to resize!"
            )

            # Build static overlay preview for Zoom mode / Static display
            overlay_preview = review_img.copy()
            draw_ctx = ImageDraw.Draw(overlay_preview)
            for app_b in approved_boxes:
                app_cls = app_b.get("class", "object")
                cid = class_map.get(app_cls, "?")
                draw_box_with_label(
                    draw_ctx,
                    app_b["box"],
                    label_text=f"[ID {cid}] {app_cls} (Approved)",
                    color="#2b7a5a",
                    line_width=max(2, overlay_preview.width // 350),
                )
            for p_idx, cand in enumerate(review_candidates):
                cand_cls = cand.get("class", "object")
                cid = class_map.get(cand_cls, "?")
                score_str = f" ({cand['score']:.2f})" if isinstance(cand.get("score"), (int, float)) else ""
                draw_box_with_label(
                    draw_ctx,
                    cand["box"],
                    label_text=f"P#{p_idx + 1} [ID {cid}] {cand_cls}{score_str}",
                    color="#e06b45",
                    line_width=max(2, overlay_preview.width // 350),
                )

            scale_factor = min(1.0, 760.0 / review_img.width)
            display_size = (int(review_img.width * scale_factor), int(review_img.height * scale_factor))
            disp_cache_key = f"_disp_{review_name}_{display_size[0]}_{display_size[1]}"
            if disp_cache_key not in st.session_state:
                st.session_state[disp_cache_key] = review_img.resize(display_size)
            clean_display_img = st.session_state[disp_cache_key]

            # Build initial interactive drawing objects for canvas
            canvas_objects = []
            for a_idx, app_b in enumerate(approved_boxes):
                app_cls = app_b.get("class", "object")
                cid = class_map.get(app_cls, "?")
                bx1, by1, bx2, by2 = app_b["box"]
                canvas_objects.append({
                    "type": "LabeledRect",
                    "label": f"Approved #{a_idx + 1}: [ID {cid}] {app_cls}",
                    "left": int(round(bx1 * scale_factor)),
                    "top": int(round(by1 * scale_factor)),
                    "width": int(round((bx2 - bx1) * scale_factor)),
                    "height": int(round((by2 - by1) * scale_factor)),
                    "originX": "left",
                    "originY": "top",
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "stroke": "#2b7a5a",
                    "strokeWidth": 3,
                    "fill": "rgba(43, 122, 90, 0.18)",
                    "lockRotation": True,
                    "strokeUniform": True,
                })

            for p_idx, cand in enumerate(review_candidates):
                cand_cls = cand.get("class", "object")
                cid = class_map.get(cand_cls, "?")
                score_str = f" ({cand['score']:.2f})" if isinstance(cand.get("score"), (int, float)) else ""
                bx1, by1, bx2, by2 = cand["box"]
                canvas_objects.append({
                    "type": "LabeledRect",
                    "label": f"Proposal #{p_idx + 1}: [ID {cid}] {cand_cls}{score_str}",
                    "left": int(round(bx1 * scale_factor)),
                    "top": int(round(by1 * scale_factor)),
                    "width": int(round((bx2 - bx1) * scale_factor)),
                    "height": int(round((by2 - by1) * scale_factor)),
                    "originX": "left",
                    "originY": "top",
                    "scaleX": 1,
                    "scaleY": 1,
                    "angle": 0,
                    "stroke": "#e06b45",
                    "strokeWidth": 3,
                    "fill": "rgba(224, 107, 69, 0.22)",
                    "lockRotation": True,
                    "strokeUniform": True,
                })

            initial_drawing_data = {"objects": canvas_objects}

            # Toolbar & Interaction Mode Selector with Active Class Assignment
            toolbar_col1, toolbar_col2, toolbar_col3 = st.columns([1.8, 1.2, 1.2])

            canvas_tool = toolbar_col1.radio(
                "Canvas Interaction Tool",
                [
                    "🖐️ Direct Edit Mode (Move & resize boxes directly on image)",
                    "✏️ Draw New Box Mode (Click & drag with mouse to draw)",
                    "🔍 Static Image Preview",
                ],
                index=0,
                key=f"canvas_tool_{review_name}",
            )

            # Available classes across labels, references, pending proposals
            cand_classes = {c.get("class") for c in review_candidates if c.get("class")}
            app_classes = {b.get("class") for b in approved_boxes if b.get("class")}
            available_classes = sorted(base_known_classes | cand_classes | app_classes)
            if not available_classes:
                available_classes = ["object"]

            active_select_cls = toolbar_col2.selectbox(
                "Class for Added Box",
                available_classes,
                format_func=lambda c: f"ID {class_map.get(c, '?')}: {c}",
                key=f"active_draw_cls_{review_name}",
                help="Class that will be assigned to any box drawn or manually added.",
            )

            active_custom_cls = toolbar_col3.text_input(
                "Or new class name",
                placeholder="e.g. helmet, pillow",
                key=f"active_custom_cls_{review_name}",
                help="Type a custom class name if not in the dropdown.",
            ).strip()

            active_draw_class = active_custom_cls if active_custom_cls else active_select_cls
            active_class_id = class_map.get(active_draw_class, len(class_map))

            if "canvas_revision" not in st.session_state:
                st.session_state.canvas_revision = {}
            canvas_rev = st.session_state.canvas_revision.get(review_name, 0)

            app_updates, prop_updates, new_drawn = {}, {}, []
            drawn_box_coords = None

            if canvas_tool.startswith("🔍 Static"):
                st.image(overlay_preview, use_container_width=True)
            elif st_canvas is not None:
                is_drawing = canvas_tool.startswith("✏️ Draw New")
                drawing_mode = "labeled_rect" if is_drawing else "transform"
                canvas_label = f"[ID {active_class_id}] {active_draw_class}" if is_drawing else ""
                canvas_key = f"review_canvas_{review_name}_{canvas_rev}_{drawing_mode}" + (f"_{active_draw_class}" if is_drawing else "")

                canvas_result = st_canvas(
                    background_image=clean_display_img,
                    drawing_mode=drawing_mode,
                    stroke_color="#3b82f6" if is_drawing else "#e06b45",
                    stroke_width=3,
                    fill_color="rgba(59, 130, 246, 0.22)" if is_drawing else "rgba(224, 107, 69, 0.22)",
                    height=display_size[1],
                    width=display_size[0],
                    initial_drawing=initial_drawing_data,
                    label=canvas_label,
                    key=canvas_key,
                )

                if canvas_result and canvas_result.json_data:
                    raw_objs = canvas_result.json_data.get("objects", [])
                    app_updates, prop_updates, new_drawn = parse_canvas_objects(
                        raw_objs, display_size[0], display_size[1], review_img.width, review_img.height
                    )
                    for b in new_drawn:
                        if not b.get("class"):
                            b["class"] = active_draw_class
                    if new_drawn:
                        drawn_box_coords = new_drawn[-1]["box"]
            else:
                st.error("Interactive canvas requires `streamlit-drawable-canvas`.")
                st.image(overlay_preview, use_container_width=True)

            # Check if any on-image modifications were made by the user
            canvas_has_returned_data = bool(
                canvas_result and canvas_result.json_data and "objects" in canvas_result.json_data
            )
            proposals_modified = any(
                i in prop_updates and prop_updates[i] != cand["box"]
                for i, cand in enumerate(review_candidates)
            )
            approved_modified = any(
                i in app_updates and app_updates[i] != app_b["box"]
                for i, app_b in enumerate(approved_boxes)
            )
            proposals_deleted_on_canvas = (
                canvas_has_returned_data
                and len(review_candidates) > 0
                and len(prop_updates) < len(review_candidates)
            )
            new_boxes_drawn = len(new_drawn) > 0
            has_canvas_changes = proposals_modified or approved_modified or proposals_deleted_on_canvas or new_boxes_drawn

            if has_canvas_changes:
                with st.container(border=True):
                    st.markdown("##### ⚡ On-Image Edits Detected")
                    if new_boxes_drawn:
                        drawn_summary = ", ".join(f"**`{b.get('class', active_draw_class)}`**" for b in new_drawn)
                        st.info(
                            f"✨ **{len(new_drawn)} new bounding box(es) drawn** on the canvas, assigned to class: {drawn_summary}. "
                            f"Click **'Save On-Image Edits to Proposals'** or **'Approve All from Canvas'** to save them."
                        )
                    else:
                        st.info(
                            "You moved, resized, or removed boxes directly on the image canvas! "
                            "Click **'Save On-Image Edits to Proposals'** to apply these coordinates to your proposals, or **'Approve All from Canvas'** to accept them into the dataset."
                        )
                    ed_c1, ed_c2, ed_c3 = st.columns([1.5, 1.5, 1])

                    if ed_c1.button("💾 Save On-Image Edits to Proposals", type="primary", key=f"save_canvas_edits_{review_name}"):
                        new_pending = []
                        for i, cand in enumerate(review_candidates):
                            if i in prop_updates:
                                new_box = prop_updates[i]
                                if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
                                    cand["box"] = new_box
                                new_pending.append(cand)
                            else:
                                st.session_state.rejections.append(1)

                        for idx, b in enumerate(new_drawn):
                            b_cls = b.get("class") or active_draw_class
                            new_pending.append({
                                "class": b_cls,
                                "box": b["box"],
                                "score": 1.0,
                                "source": "manual",
                                "id": f"manual_{time.time()}_{idx}",
                            })

                        st.session_state.pending[review_name] = new_pending

                        for i, app_b in enumerate(approved_boxes):
                            if i in app_updates:
                                new_box = app_updates[i]
                                if new_box[2] > new_box[0] and new_box[3] > new_box[1]:
                                    app_b["box"] = new_box
                                    st.session_state.refs[app_b["class"]].append(crop_box(review_img, new_box))

                        st.session_state.canvas_revision[review_name] = canvas_rev + 1
                        # Invalidate cached embeddings: references may have changed.
                        if st.session_state.get("matcher") is not None:
                            st.session_state.matcher.invalidate_reference_cache()
                        persist_active_workspace()
                        st.success("Applied on-image edits successfully!")
                        st.rerun()

                    if ed_c2.button("✅ Approve All from Canvas", type="secondary", key=f"approve_canvas_all_{review_name}"):
                        for i, cand in enumerate(review_candidates):
                            if i in prop_updates:
                                box = prop_updates[i]
                                if box[2] > box[0] and box[3] > box[1]:
                                    cand["box"] = box
                                approved_entry = {**cand, "source": "auto"}
                                st.session_state.labels[review_name].append(approved_entry)
                                st.session_state.refs[cand["class"]].append(crop_box(review_img, cand["box"]))
                            else:
                                st.session_state.rejections.append(1)

                        for idx, b in enumerate(new_drawn):
                            b_cls = b.get("class") or active_draw_class
                            approved_entry = {"class": b_cls, "box": b["box"], "source": "manual"}
                            st.session_state.labels[review_name].append(approved_entry)
                            st.session_state.refs[b_cls].append(crop_box(review_img, b["box"]))

                        st.session_state.pending[review_name] = []
                        st.session_state.canvas_revision[review_name] = canvas_rev + 1
                        # Invalidate cached embeddings: new references were added.
                        if st.session_state.get("matcher") is not None:
                            st.session_state.matcher.invalidate_reference_cache()
                        persist_active_workspace()
                        st.success("Approved all canvas proposals!")
                        st.rerun()

                    if ed_c3.button("↺ Reset Canvas", key=f"reset_canvas_{review_name}"):
                        st.session_state.canvas_revision[review_name] = canvas_rev + 1
                        st.rerun()

            # -----------------------------------------------------------------
            # Action Section: Add Newer Box
            # -----------------------------------------------------------------
            with st.container(border=True):
                st.markdown("##### ➕ Add Newer Bounding Box to this Image")
                st.caption(f"Assigned class: **`[ID {active_class_id}] {active_draw_class}`** (change class in toolbar above if needed)")

                # If a box was drawn on canvas, show its coordinates and add buttons
                if drawn_box_coords:
                    st.success(
                        f"🎯 **Drawn box detected on canvas:** `[{drawn_box_coords[0]}, {drawn_box_coords[1]}, {drawn_box_coords[2]}, {drawn_box_coords[3]}]` · "
                        f"Assigned Class: **`[ID {active_class_id}] {active_draw_class}`**"
                    )
                    add_btn_c1, add_btn_c2 = st.columns(2)
                    if add_btn_c1.button(f"✅ Add as Approved Box ({active_draw_class})", type="primary", key=f"add_drawn_app_{review_name}"):
                        new_entry = {"class": active_draw_class, "box": drawn_box_coords, "source": "manual"}
                        st.session_state.labels[review_name].append(new_entry)
                        st.session_state.refs[active_draw_class].append(crop_box(review_img, drawn_box_coords))
                        st.session_state.canvas_revision[review_name] = canvas_rev + 1
                        # Invalidate cache: a new reference crop was added.
                        if st.session_state.get("matcher") is not None:
                            st.session_state.matcher.invalidate_reference_cache()
                        persist_active_workspace()
                        st.success(f"Added approved box for '{active_draw_class}'!")
                        st.rerun()

                    if add_btn_c2.button(f"🟠 Add to Review Proposals ({active_draw_class})", key=f"add_drawn_prop_{review_name}"):
                        new_prop = {"class": active_draw_class, "box": drawn_box_coords, "score": 1.0, "source": "manual", "id": f"manual_{time.time()}"}
                        st.session_state.pending[review_name].append(new_prop)
                        st.session_state.canvas_revision[review_name] = canvas_rev + 1
                        persist_active_workspace()
                        st.success(f"Added review proposal for '{active_draw_class}'!")
                        st.rerun()
                else:
                    st.caption("💡 Switch toolbar above to **'✏️ Draw New Box Mode'** to draw directly with mouse, or enter coordinates below.")

                with st.expander("Or enter coordinates manually"):
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    mx1 = mc1.number_input("Left", 0, review_img.width, 0, key=f"man_x1_{review_name}")
                    my1 = mc2.number_input("Top", 0, review_img.height, 0, key=f"man_y1_{review_name}")
                    mx2 = mc3.number_input("Right", 0, review_img.width, min(review_img.width, 100), key=f"man_x2_{review_name}")
                    my2 = mc4.number_input("Bottom", 0, review_img.height, min(review_img.height, 100), key=f"man_y2_{review_name}")
                    if mc5.button(f"Add as '{active_draw_class}'", key=f"add_manual_btn_{review_name}"):
                        if mx2 > mx1 and my2 > my1:
                            mcoords = [int(mx1), int(my1), int(mx2), int(my2)]
                            st.session_state.labels[review_name].append({"class": active_draw_class, "box": mcoords, "source": "manual"})
                            st.session_state.refs[active_draw_class].append(crop_box(review_img, mcoords))
                            st.session_state.canvas_revision[review_name] = canvas_rev + 1
                            # Invalidate cache: a new reference crop was added.
                            if st.session_state.get("matcher") is not None:
                                st.session_state.matcher.invalidate_reference_cache()
                            persist_active_workspace()
                            st.success(f"Added manual box for '{active_draw_class}'!")
                            st.rerun()
                        else:
                            st.error("Right must be > Left and Bottom must be > Top.")

            # -----------------------------------------------------------------
            # Section: Pending Candidate Proposals (Edit, Approve, Delete)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown(f"#### 🟠 Pending Candidate Proposals ({len(review_candidates)})")

            if not review_candidates:
                st.caption("No pending proposals for this image. All candidates have been approved or rejected.")
            else:
                bulk_col1, bulk_col2 = st.columns(2)
                if bulk_col1.button("✅ Approve All Remaining Proposals", type="secondary", key=f"bulk_app_{review_name}"):
                    for cand in review_candidates:
                        approved_entry = {**cand, "source": "auto"}
                        st.session_state.labels[review_name].append(approved_entry)
                        st.session_state.refs[cand["class"]].append(crop_box(review_img, cand["box"]))
                    st.session_state.pending[review_name] = []
                    st.session_state.canvas_revision[review_name] = canvas_rev + 1
                    # Invalidate cache: new reference crops were added.
                    if st.session_state.get("matcher") is not None:
                        st.session_state.matcher.invalidate_reference_cache()
                    persist_active_workspace()
                    st.rerun()

                if bulk_col2.button("🗑️ Reject / Delete All Proposals", key=f"bulk_rej_{review_name}"):
                    st.session_state.rejections.extend([1] * len(review_candidates))
                    st.session_state.pending[review_name] = []
                    st.session_state.canvas_revision[review_name] = canvas_rev + 1
                    # Rejected candidates do not become references; no cache invalidation needed.
                    persist_active_workspace()
                    st.rerun()

                for idx, candidate in enumerate(list(review_candidates)):
                    cand_id = candidate.get("id") or f"{candidate['box'][0]}_{candidate['box'][1]}_{idx}"
                    current_cand_box = prop_updates.get(idx, candidate["box"])
                    is_on_image_edited = (idx in prop_updates and prop_updates[idx] != candidate["box"])

                    with st.container(border=True):
                        col_hdr, col_cls, col_save, col_app, col_rej = st.columns([2.5, 1.5, 1, 1, 1])
                        score_display = f"{candidate['score']:.2f}" if isinstance(candidate.get("score"), (int, float)) else "—"
                        cand_cid = class_map.get(candidate['class'], "?")

                        edit_badge = f" · <span style='color: #2563eb; font-weight: 600;'>[✏️ Moved on image]</span>" if is_on_image_edited else ""
                        col_hdr.markdown(
                            f"**Box #{idx + 1}** · **`[ID {cand_cid}] {candidate['class']}`** · Score: `{score_display}`{edit_badge}",
                            unsafe_allow_html=True,
                        )

                        cand_class = candidate["class"]
                        class_options = sorted(base_known_classes | {cand_class})
                        default_idx = class_options.index(cand_class) if cand_class in class_options else 0

                        chosen_class = col_cls.selectbox(
                            "Class",
                            class_options,
                            index=default_idx,
                            format_func=lambda c: f"ID {class_map.get(c, '?')}: {c}",
                            key=f"reclass-{review_name}-{cand_id}",
                            label_visibility="collapsed",
                        )

                        # Inline Coordinate Editors (synced with on-image coordinates)
                        with st.expander(f"✏️ Coordinate Details: [{current_cand_box[0]}, {current_cand_box[1]}, {current_cand_box[2]}, {current_cand_box[3]}]", expanded=False):
                            coord_cols = st.columns(4)
                            x1_val = coord_cols[0].number_input(
                                "Left (x1)", 0, review_img.width, int(current_cand_box[0]), key=f"x1-{review_name}-{cand_id}-{canvas_rev}"
                            )
                            y1_val = coord_cols[1].number_input(
                                "Top (y1)", 0, review_img.height, int(current_cand_box[1]), key=f"y1-{review_name}-{cand_id}-{canvas_rev}"
                            )
                            x2_val = coord_cols[2].number_input(
                                "Right (x2)", 0, review_img.width, int(current_cand_box[2]), key=f"x2-{review_name}-{cand_id}-{canvas_rev}"
                            )
                            y2_val = coord_cols[3].number_input(
                                "Bottom (y2)", 0, review_img.height, int(current_cand_box[3]), key=f"y2-{review_name}-{cand_id}-{canvas_rev}"
                            )

                        adjusted_box = [int(x1_val), int(y1_val), int(x2_val), int(y2_val)]

                        # 1. Edit & Save changes to proposal
                        if col_save.button("💾 Save", key=f"save-{review_name}-{cand_id}", help="Save updated class and coordinates to this proposal"):
                            if adjusted_box[2] > adjusted_box[0] and adjusted_box[3] > adjusted_box[1]:
                                st.session_state.pending[review_name][idx]["class"] = chosen_class
                                st.session_state.pending[review_name][idx]["box"] = adjusted_box
                                st.session_state.canvas_revision[review_name] = canvas_rev + 1
                                persist_active_workspace()
                                st.rerun()
                            else:
                                st.error("Invalid coordinates: Right must be > Left and Bottom must be > Top.")

                        # 2. Approve
                        if col_app.button("✅ Approve", key=f"approve-{review_name}-{cand_id}", type="primary", help="Approve this box into the dataset"):
                            if adjusted_box[2] > adjusted_box[0] and adjusted_box[3] > adjusted_box[1]:
                                approved_entry = {
                                    **candidate,
                                    "class": chosen_class,
                                    "box": adjusted_box,
                                    "source": "auto",
                                }
                                st.session_state.labels[review_name].append(approved_entry)
                                st.session_state.refs[chosen_class].append(crop_box(review_img, approved_entry["box"]))
                                st.session_state.pending[review_name].pop(idx)
                                st.session_state.canvas_revision[review_name] = canvas_rev + 1
                                # Invalidate cache: a new reference crop was added.
                                if st.session_state.get("matcher") is not None:
                                    st.session_state.matcher.invalidate_reference_cache()
                                persist_active_workspace()
                                st.rerun()

                        # 3. Delete / Reject
                        if col_rej.button("🗑️ Delete", key=f"reject-{review_name}-{cand_id}", help="Delete and reject this proposal"):
                            st.session_state.pending[review_name].pop(idx)
                            st.session_state.rejections.append(1)
                            st.session_state.canvas_revision[review_name] = canvas_rev + 1
                            persist_active_workspace()
                            st.rerun()

            # -----------------------------------------------------------------
            # Section: Approved Boxes on Image (Edit & Delete)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.markdown(f"#### 🟢 Approved Annotations on this Image ({len(approved_boxes)})")

            if not approved_boxes:
                st.caption("No approved boxes on this image yet.")
            else:
                for a_idx, app_item in enumerate(list(approved_boxes)):
                    app_box_id = f"app_{a_idx}_{app_item['box'][0]}_{app_item['box'][1]}"
                    current_app_box = app_updates.get(a_idx, app_item["box"])
                    app_is_edited = (a_idx in app_updates and app_updates[a_idx] != app_item["box"])

                    with st.container(border=True):
                        ahdr_col, aedit_col, adel_col = st.columns([3, 1, 1])
                        a_cls = app_item.get("class", "object")
                        a_cid = class_map.get(a_cls, "?")
                        app_edit_badge = f" · <span style='color: #2563eb; font-weight: 600;'>[✏️ Moved on image]</span>" if app_is_edited else ""
                        ahdr_col.markdown(
                            f"**`[ID {a_cid}] {a_cls}`** · Box: `[{current_app_box[0]}, {current_app_box[1]}, {current_app_box[2]}, {current_app_box[3]}]` · *{app_item.get('source', 'manual')}*{app_edit_badge}",
                            unsafe_allow_html=True,
                        )

                        with st.expander(f"✏️ Edit Approved Box #{a_idx + 1}"):
                            ec_cls, ec_x1, ec_y1, ec_x2, ec_y2, ec_btn = st.columns([2, 1, 1, 1, 1, 1])
                            app_class_options = sorted(base_known_classes | {a_cls})
                            app_def_idx = app_class_options.index(a_cls) if a_cls in app_class_options else 0
                            new_app_cls = ec_cls.selectbox(
                                "Class",
                                app_class_options,
                                index=app_def_idx,
                                format_func=lambda c: f"ID {class_map.get(c, '?')}: {c}",
                                key=f"edit_app_cls_{review_name}_{app_box_id}",
                            )
                            ax1 = ec_x1.number_input("Left", 0, review_img.width, int(current_app_box[0]), key=f"eax1_{review_name}_{app_box_id}-{canvas_rev}")
                            ay1 = ec_y1.number_input("Top", 0, review_img.height, int(current_app_box[1]), key=f"eay1_{review_name}_{app_box_id}-{canvas_rev}")
                            ax2 = ec_x2.number_input("Right", 0, review_img.width, int(current_app_box[2]), key=f"eax2_{review_name}_{app_box_id}-{canvas_rev}")
                            ay2 = ec_y2.number_input("Bottom", 0, review_img.height, int(current_app_box[3]), key=f"eay2_{review_name}_{app_box_id}-{canvas_rev}")

                            if ec_btn.button("💾 Save", key=f"save_app_{review_name}_{app_box_id}"):
                                if ax2 > ax1 and ay2 > ay1:
                                    updated_box = [int(ax1), int(ay1), int(ax2), int(ay2)]
                                    st.session_state.labels[review_name][a_idx]["class"] = new_app_cls
                                    st.session_state.labels[review_name][a_idx]["box"] = updated_box
                                    st.session_state.refs[new_app_cls].append(crop_box(review_img, updated_box))
                                    st.session_state.canvas_revision[review_name] = canvas_rev + 1
                                    # Invalidate cache: an updated reference crop was added.
                                    if st.session_state.get("matcher") is not None:
                                        st.session_state.matcher.invalidate_reference_cache()
                                    persist_active_workspace()
                                    st.success("Updated approved box!")
                                    st.rerun()
                                else:
                                    st.error("Right must be > Left and Bottom must be > Top.")

                        if adel_col.button("🗑️ Delete Box", key=f"del_app_{review_name}_{app_box_id}"):
                            st.session_state.labels[review_name].pop(a_idx)
                            st.session_state.canvas_revision[review_name] = canvas_rev + 1
                            persist_active_workspace()
                            st.rerun()

    # -------------------------------------------------------------------------
    # Tab 3: Batch Folder Predict & Visual Inspection
    # -------------------------------------------------------------------------
    with tab_batch:
        st.markdown("### 📂 Batch Folder Predict & Visual Inspection")
        st.caption(
            "Upload a folder of images or specify a local folder path. The accelerated vision model "
            "will scan all images, propose bounding boxes, and display the predicted boxes in a visual gallery."
        )

        accel_badge = (
            f"⚡ **Compute Acceleration:** CUDA GPU Active ({device_status})"
            if resolved_device == "cuda"
            else f"ℹ️ **Compute Device:** CPU Mode ({device_status})"
        )
        st.markdown(
            f'<div style="padding: 6px 12px; border-radius: 6px; background: {"#e6f4ea" if resolved_device == "cuda" else "#f3f4f6"}; '
            f'border: 1px solid {"#ceead6" if resolved_device == "cuda" else "#e5e7eb"}; '
            f'color: {"#137333" if resolved_device == "cuda" else "#374151"}; font-size: 13px; font-weight: 600; margin-bottom: 14px;">'
            f'{accel_badge}</div>',
            unsafe_allow_html=True,
        )

        # 1. Image Source Selection
        st.markdown("#### 1️⃣ Select Image Source")
        source_mode = st.radio(
            "Folder Input Method",
            ["Local Folder Path on Disk", "Upload Images via Browser", "Use Current Workspace Images"],
            horizontal=True,
            key="batch_folder_input_mode",
        )

        if source_mode == "Local Folder Path on Disk":
            st.markdown("Select a folder using the **file explorer pop-up window** or paste the path below:")
            col_browse_btn, col_browse_tip = st.columns([1.5, 2])
            with col_browse_btn:
                if st.button("🗂️ Select Folder (Pop-up Window)...", type="primary", use_container_width=True, key="btn_popup_folder_picker"):
                    picked_dir = open_folder_picker(initial_dir=st.session_state.get("batch_last_folder_path", ""))
                    if picked_dir:
                        st.session_state.batch_last_folder_path = picked_dir
                        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
                        found_files = [
                            f for f in sorted(os.listdir(picked_dir))
                            if os.path.splitext(f)[1].lower() in valid_exts
                        ]
                        if not found_files:
                            st.warning(f"No JPG, PNG, or WEBP images found in '{picked_dir}'.")
                        else:
                            loaded_imgs = {}
                            for fname in found_files:
                                full_p = os.path.join(picked_dir, fname)
                                try:
                                    with Image.open(full_p) as im:
                                        loaded_imgs[fname] = im.convert("RGB")
                                except Exception:
                                    pass
                            st.session_state.batch_folder_images = loaded_imgs
                            st.session_state.batch_folder_results = {}
                            st.success(f"Selected and loaded {len(loaded_imgs)} image(s) from '{picked_dir}'!")
                            st.rerun()
            with col_browse_tip:
                st.caption("Opens a native file explorer pop-up window so you can simply click and choose any folder.")

            col_path, col_btn = st.columns([3, 1])
            with col_path:
                disk_folder_input = st.text_input(
                    "Or Type / Paste Folder Path",
                    value=st.session_state.get("batch_last_folder_path", ""),
                    placeholder=r"e.g. C:\Users\sarim\Pictures\dataset",
                    help="Click 'Select Folder' above or enter the full path to a local directory containing images.",
                    key="batch_folder_disk_input",
                )
            with col_btn:
                st.write("")
                st.write("")
                if st.button("📂 Load Path", use_container_width=True, type="secondary"):
                    clean_path = disk_folder_input.strip().strip('"').strip("'")
                    if not clean_path or not os.path.isdir(clean_path):
                        st.error(f"Directory not found: '{clean_path}'")
                    else:
                        st.session_state.batch_last_folder_path = clean_path
                        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
                        found_files = [
                            f for f in sorted(os.listdir(clean_path))
                            if os.path.splitext(f)[1].lower() in valid_exts
                        ]
                        if not found_files:
                            st.warning(f"No JPG, PNG, or WEBP images found in '{clean_path}'.")
                        else:
                            loaded_imgs = {}
                            for fname in found_files:
                                full_p = os.path.join(clean_path, fname)
                                try:
                                    with Image.open(full_p) as im:
                                        loaded_imgs[fname] = im.convert("RGB")
                                except Exception:
                                    pass
                            st.session_state.batch_folder_images = loaded_imgs
                            st.session_state.batch_folder_results = {}
                            st.success(f"Successfully loaded {len(loaded_imgs)} image(s) from folder!")
                            st.rerun()

        elif source_mode == "Upload Images via Browser":
            uploaded_folder_files = st.file_uploader(
                "Select images or drag folder contents",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="batch_folder_uploader",
                help="Select all image files from your folder to load them into the batch predictor.",
            )
            if uploaded_folder_files:
                new_batch = {}
                for u in uploaded_folder_files:
                    try:
                        u.seek(0)
                        new_batch[u.name] = Image.open(u).convert("RGB")
                    except Exception:
                        pass
                if len(new_batch) != len(st.session_state.batch_folder_images):
                    st.session_state.batch_folder_images = new_batch
                    st.session_state.batch_folder_results = {}
                    st.success(f"Loaded {len(new_batch)} uploaded image(s)!")

        else:
            if st.button(f"📥 Copy All {len(image_names)} Workspace Images into Batch", use_container_width=True):
                st.session_state.batch_folder_images = {k: v.copy() for k, v in st.session_state.images.items()}
                st.session_state.batch_folder_results = {}
                st.success(f"Imported {len(image_names)} images from active workspace into batch predictor!")
                st.rerun()

        # Display current batch status
        curr_batch_count = len(st.session_state.batch_folder_images)
        if curr_batch_count == 0:
            st.info("👆 Please select or upload a folder of images above to begin batch prediction.")
        else:
            c_info1, c_info2 = st.columns([3, 1])
            with c_info1:
                st.write(f"📁 **Active Batch:** **{curr_batch_count}** images queued for prediction.")
            with c_info2:
                if st.button("🗑️ Clear Batch", use_container_width=True):
                    st.session_state.batch_folder_images = {}
                    st.session_state.batch_folder_results = {}
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 2️⃣ Prediction Settings & Execution")

            pred_col1, pred_col2 = st.columns([1, 1])
            with pred_col1:
                batch_model_choice = st.selectbox(
                    "Model Engine",
                    MODEL_CHOICES,
                    index=MODEL_CHOICES.index(model_choice) if model_choice in MODEL_CHOICES else 0,
                    key="batch_model_choice_select",
                )
            with pred_col2:
                if batch_model_choice == MODEL_CLIP_REFERENCE:
                    batch_thresh = st.slider(
                        "Similarity Threshold",
                        min_value=0.10,
                        max_value=0.95,
                        value=float(persisted_s.get("clip_threshold", 0.45)),
                        step=0.01,
                        key="batch_ref_thresh_slider",
                        help="Similarity cutoff. Higher values require closer visual match.",
                    )
                else:
                    batch_thresh = st.slider(
                        "Box Confidence Threshold",
                        min_value=0.05,
                        max_value=0.95,
                        value=float(persisted_s.get("dino_box_thresh", 0.35)),
                        step=0.01,
                        key="batch_text_thresh_slider",
                    )

            if batch_model_choice == MODEL_CLIP_REFERENCE:
                total_refs_count = sum(len(crops) for crops in st.session_state.refs.values())
                if total_refs_count == 0:
                    st.warning("⚠️ No visual references available in the reference library. Please teach at least one reference crop in Tab 01 before scanning.")
                    can_predict = False
                else:
                    active_classes = sorted([k for k, v in st.session_state.refs.items() if v])
                    st.caption(f"🎯 Target visual classes ({len(active_classes)}): **{', '.join(active_classes)}** ({total_refs_count} total reference crops).")
                    st.caption("🚗 **Single Most Accurate Box Mode:** Strictly returns the single highest-scoring car box per image.")
                    can_predict = True
            else:
                prompt_input = st.text_input(
                    "Target Class Prompt(s)",
                    value=", ".join(get_class_id_map().keys()) or "car",
                    key="batch_text_prompt_input",
                )
                can_predict = bool(prompt_input.strip())

            if st.button(
                f"⚡ Run Predictions for All {curr_batch_count} Images",
                type="primary",
                disabled=not can_predict,
                use_container_width=True,
                key="btn_run_batch_predictions",
            ):
                with st.spinner("Preparing model for batch prediction..."):
                    if batch_model_choice == MODEL_CLIP_REFERENCE:
                        if (
                            st.session_state.get("matcher") is None
                            or st.session_state.get("matcher_device") != device_choice
                            or not hasattr(st.session_state.get("matcher"), "_proposals_opencv")
                        ):
                            st.session_state.matcher = ReferenceMatcher(device_choice)
                            st.session_state.matcher_device = device_choice
                        engine = st.session_state.matcher
                    else:
                        if st.session_state.get("detector") is None or st.session_state.get("detector_device") != device_choice:
                            st.session_state.detector = GroundingDinoDetector(device=device_choice)
                            st.session_state.detector_device = device_choice
                        engine = st.session_state.detector

                p_bar = st.progress(0.0)
                status_txt = st.empty()
                results = {}
                batch_start = time.perf_counter()
                total_found_boxes = 0

                batch_items = list(st.session_state.batch_folder_images.items())
                for b_idx, (b_fname, b_img) in enumerate(batch_items):
                    status_txt.markdown(f"🔍 Predicting **{b_fname}** ({b_idx + 1}/{len(batch_items)}) · Found **{total_found_boxes}** box(es) so far...")
                    t_img_start = time.perf_counter()

                    if batch_model_choice == MODEL_CLIP_REFERENCE:
                        chosen_prop_method = (
                            "sam"
                            if persisted_s.get("proposal_method") == PROPOSAL_SAM
                            else "opencv"
                        )
                        predict_kwargs = dict(
                            threshold=batch_thresh,
                            limit=int(persisted_s.get("detail_proposals", 240)),
                            margin=0.05,
                            car_nms_iou=0.50,
                        )
                        if "proposal_method" in inspect.signature(engine.predict).parameters:
                            predict_kwargs["proposal_method"] = chosen_prop_method
                        preds = engine.predict(
                            b_img,
                            st.session_state.refs,
                            **predict_kwargs,
                        )
                        preds = filter_one_car_per_image(preds)
                    else:
                        preds = engine.predict(
                            b_img,
                            prompt=prompt_input,
                            box_threshold=batch_thresh,
                            text_threshold=float(persisted_s.get("dino_text_thresh", 0.25)),
                            target_size=None,
                        )

                    t_elapsed = time.perf_counter() - t_img_start

                    # Create annotated visual copy with rendered bounding boxes
                    ann_img = b_img.copy()
                    draw_c = ImageDraw.Draw(ann_img)
                    for p in preds:
                        bx = p["box"]
                        cl = p.get("class", "object")
                        sc = p.get("score")
                        badge_txt = f"{cl} · {sc:.2f}" if isinstance(sc, (int, float)) else str(cl)
                        draw_box_with_label(draw_c, bx, badge_txt, color="#00e676", text_color="#000000", line_width=4)

                    for p_i, p_item in enumerate(preds):
                        p_item["id"] = f"batch_{b_fname}_{p_i}_{time.time()}"

                    results[b_fname] = {
                        "image": b_img,
                        "annotated_image": ann_img,
                        "predictions": preds,
                        "time": t_elapsed,
                    }
                    total_found_boxes += len(preds)
                    p_bar.progress((b_idx + 1) / len(batch_items))

                total_batch_duration = time.perf_counter() - batch_start
                status_txt.empty()
                p_bar.empty()
                st.session_state.batch_folder_results = results

                avg_per_img = total_batch_duration / max(1, len(batch_items))
                st.success(
                    f"🎉 Successfully analyzed {len(batch_items)} image(s) in {total_batch_duration:.1f}s "
                    f"({avg_per_img:.2f}s/image)! Identified {total_found_boxes} bounding box(es)."
                )

        # ---------------------------------------------------------------------
        # 3️⃣ Visual Results Gallery & Bounding Box Inspection
        # ---------------------------------------------------------------------
        b_results = st.session_state.get("batch_folder_results", {})
        if b_results:
            st.markdown("---")
            st.markdown("#### 3️⃣ Predicted Bounding Boxes Gallery")

            res_total_imgs = len(b_results)
            res_imgs_with_boxes = sum(1 for r in b_results.values() if r["predictions"])
            res_total_boxes = sum(len(r["predictions"]) for r in b_results.values())
            res_total_time = sum(r["time"] for r in b_results.values())
            res_avg_time = res_total_time / max(1, res_total_imgs)

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.markdown(f'<div class="metric-card"><span>Total Scanned</span><strong>{res_total_imgs}</strong></div>', unsafe_allow_html=True)
            sm2.markdown(f'<div class="metric-card"><span>Images with Boxes</span><strong>{res_imgs_with_boxes}</strong></div>', unsafe_allow_html=True)
            sm3.markdown(f'<div class="metric-card"><span>Total Detected Boxes</span><strong>{res_total_boxes}</strong></div>', unsafe_allow_html=True)
            sm4.markdown(f'<div class="metric-card"><span>Avg Inference Speed</span><strong>{res_avg_time:.2f}s</strong></div>', unsafe_allow_html=True)

            col_filter, col_cols = st.columns([2, 1])
            with col_filter:
                filter_choice = st.radio(
                    "Display filter",
                    ["All Images", "Only with Detected Boxes", "Only without Detections"],
                    horizontal=True,
                    key="batch_gallery_filter",
                )
            with col_cols:
                grid_cols = st.selectbox("Gallery Columns", [2, 3, 4], index=1, key="batch_gallery_grid_cols")

            if filter_choice == "Only with Detected Boxes":
                filtered_items = [(k, v) for k, v in b_results.items() if v["predictions"]]
            elif filter_choice == "Only without Detections":
                filtered_items = [(k, v) for k, v in b_results.items() if not v["predictions"]]
            else:
                filtered_items = list(b_results.items())

            if not filtered_items:
                st.info(f"No images match the filter '{filter_choice}'.")
            else:
                for row_start in range(0, len(filtered_items), grid_cols):
                    row_slice = filtered_items[row_start : row_start + grid_cols]
                    cols = st.columns(grid_cols)
                    for c_idx, (fname, f_data) in enumerate(row_slice):
                        with cols[c_idx]:
                            with st.container(border=True):
                                preds = f_data["predictions"]
                                box_count = len(preds)
                                st.markdown(f"**{fname}**")
                                st.image(f_data["annotated_image"], use_container_width=True)

                                if box_count > 0:
                                    for p in preds:
                                        sc_str = f" · score: {p['score']:.2f}" if "score" in p else ""
                                        st.markdown(
                                            f'<span style="display:inline-block; padding: 2px 8px; border-radius: 4px; '
                                            f'background: #e6f4ea; border: 1px solid #ceead6; color: #137333; font-size: 12px; font-weight: 600;">'
                                            f'✅ {p.get("class", "object")}{sc_str}</span>',
                                            unsafe_allow_html=True,
                                        )
                                        bx = [int(round(coord)) for coord in p["box"]]
                                        st.caption(f"Box `[{bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}]` · {bx[2]-bx[0]}×{bx[3]-bx[1]}px")
                                else:
                                    st.markdown(
                                        '<span style="display:inline-block; padding: 2px 8px; border-radius: 4px; '
                                        'background: #f3f4f6; border: 1px solid #e5e7eb; color: #6b7280; font-size: 12px;">'
                                        '⚪ No boxes detected</span>',
                                        unsafe_allow_html=True,
                                    )

            # High-Resolution Detailed Inspector
            st.markdown("---")
            st.markdown("#### 🔍 Single-Image Detail Inspector")
            inspect_name = st.selectbox(
                "Select image to inspect closely",
                list(b_results.keys()),
                key="batch_inspect_selectbox",
            )
            if inspect_name and inspect_name in b_results:
                ins_data = b_results[inspect_name]
                ins_preds = ins_data["predictions"]
                col_i1, col_i2 = st.columns([1.5, 1])
                with col_i1:
                    st.image(ins_data["annotated_image"], caption=f"{inspect_name} (Full Size)", use_container_width=True)
                with col_i2:
                    st.markdown(f"##### Image Metadata: `{inspect_name}`")
                    w_i, h_i = ins_data["image"].size
                    st.write(f"- **Dimensions:** {w_i} × {h_i} px")
                    st.write(f"- **Inference Duration:** {ins_data['time']:.2f} s")
                    st.write(f"- **Detected Objects:** {len(ins_preds)}")

                    if ins_preds:
                        st.markdown("**Predicted Bounding Boxes:**")
                        for idx_p, p in enumerate(ins_preds):
                            bx = [int(round(coord)) for coord in p["box"]]
                            st.markdown(
                                f"**Box #{idx_p + 1}:** `{p.get('class', 'object')}`\n"
                                f"- Score: `{p.get('score', 1.0):.3f}`\n"
                                f"- Coordinates: `[x1={bx[0]}, y1={bx[1]}, x2={bx[2]}, y2={bx[3]}]`\n"
                                f"- Size: `{bx[2] - bx[0]} × {bx[3] - bx[1]} px`\n"
                            )
                    else:
                        st.info("No bounding boxes detected for this image.")

            # Batch Export & Workspace Integration
            st.markdown("---")
            st.markdown("#### 4️⃣ Export & Workspace Integration")
            st.caption("Import detected boxes into your project review queue, or export all annotated images as a ZIP.")

            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                if st.button("📥 Import All into Workspace Review Queue", type="primary", use_container_width=True):
                    imported_count = 0
                    for fname, r_data in b_results.items():
                        if fname not in st.session_state.images:
                            st.session_state.images[fname] = r_data["image"]
                            st.session_state.labels[fname] = []
                            st.session_state.pending[fname] = []
                        if r_data["predictions"]:
                            st.session_state.pending[fname] = filter_one_car_per_image(r_data["predictions"])
                            imported_count += 1
                    persist_active_workspace()
                    st.success(f"Imported {len(b_results)} images and predictions for {imported_count} image(s) into Tab 02 (Review Queue)!")
                    st.rerun()

            with c_exp2:
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, r_data in b_results.items():
                        img_byte_buf = io.BytesIO()
                        r_data["annotated_image"].save(img_byte_buf, format="JPEG", quality=92)
                        zf.writestr(f"annotated_{fname}", img_byte_buf.getvalue())

                st.download_button(
                    "📦 Download Annotated Images (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="batch_annotated_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                    help="Download all batch images with bounding boxes permanently rendered on them.",
                )

    # -------------------------------------------------------------------------
    # Tab 4: Export & Metrics
    # -------------------------------------------------------------------------
    with tab_export:
        all_labels = [row for rows in st.session_state.labels.values() for row in rows]
        auto_approved = sum(r.get("source") == "auto" for r in all_labels)
        manual_approved = sum(r.get("source") == "manual" for r in all_labels)
        total_pending = sum(len(v) for v in st.session_state.pending.values())
        rejection_count = len(st.session_state.rejections)
        total_decisions = auto_approved + rejection_count + total_pending

        approval_rate = (auto_approved / total_decisions) if total_decisions > 0 else 0.0

        total_scan_seconds = sum(x["seconds"] for x in st.session_state.times)
        total_scan_images = sum(x["images"] for x in st.session_state.times)
        avg_scan_time = (total_scan_seconds / total_scan_images) if total_scan_images > 0 else 0.0

        # Section 1: Workflow Quality & Productivity Overview
        st.markdown("#### 📊 Annotation Quality & Review Metrics")
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Review Approval Rate", f"{approval_rate:.0%}" if total_decisions else "—")
        ec2.metric("Approved Auto-Boxes", auto_approved)
        ec3.metric("Approved Manual Boxes", manual_approved)
        ec4.metric("Avg Scan Speed", f"{avg_scan_time:.2f}s / image" if total_scan_images else "—")

        st.caption(
            f"**Approval Rate Formula:** Approved / (Approved + Rejected + Pending) = "
            f"{auto_approved} / ({auto_approved} + {rejection_count} + {total_pending}) = {approval_rate:.0%}. "
            "⚠️ **Notice:** This metric serves as an operator review productivity proxy measuring suggestion acceptance. "
            "It is **not** benchmark accuracy (e.g., mAP). Only approved boxes are included in final datasets."
        )

        # Section 2: Speed & Productivity Estimation
        st.markdown("---")
        st.markdown("#### ⏱️ Measured Speed & Productivity")
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            st.write(
                f"**Measured Automated Detection Time:** {total_scan_seconds:.2f} seconds across {total_scan_images} image scan(s)."
            )
            if total_scan_images > 0:
                st.write(f"**Throughput:** {avg_scan_time:.2f} seconds per image average.")
            else:
                st.caption("Run automated detection in Tab 01 to record detection timing.")

        with sc2:
            manual_baseline = st.number_input(
                "Manual labeling baseline estimate (minutes) · [User-Entered Baseline]",
                min_value=0.0,
                value=float(persisted_s.get("manual_baseline", 0.0)),
                step=1.0,
                help="Enter estimated time that would be required to draw all annotations manually without AI assistance.",
            )
            st.session_state.persisted_settings["manual_baseline"] = manual_baseline
            if manual_baseline > 0.0:
                assisted_minutes = total_scan_seconds / 60.0
                time_saved_ratio = max(0.0, 1.0 - (assisted_minutes / manual_baseline))
                st.write(
                    f"**Estimated Time Saved vs User-Entered Baseline:** **{time_saved_ratio:.0%}** "
                    f"(AI scan time: {assisted_minutes:.2f} min vs user-entered baseline: {manual_baseline:.1f} min)."
                )

        # Section 3: Ground-Truth Evaluation Benchmark (Optional)
        with st.expander("🎯 Ground-Truth Evaluation Benchmark (Precision / Recall / F1)", expanded=False):
            st.markdown(
                "Evaluate model proposals against verified ground-truth annotations using greedy IoU assignment. "
                "Unlike review approval rates, this provides true benchmark accuracy metrics when ground truth is provided."
            )
            eval_col1, eval_col2 = st.columns([1, 2])
            with eval_col1:
                eval_iou = st.slider(
                    "Benchmark IoU Threshold",
                    min_value=0.10,
                    max_value=0.90,
                    value=float(persisted_s.get("eval_iou", 0.50)),
                    step=0.05,
                    help="Minimum Intersection over Union for a prediction to count as a True Positive.",
                )
                st.session_state.persisted_settings["eval_iou"] = eval_iou

            # Find manual ground truth and auto predictions
            gt_boxes = []
            pred_boxes = []
            for img_name in image_names:
                for entry in st.session_state.labels.get(img_name, []):
                    if entry.get("source") == "manual":
                        gt_boxes.append(entry)
                    elif entry.get("source") == "auto":
                        pred_boxes.append(entry)
                for cand in st.session_state.pending.get(img_name, []):
                    pred_boxes.append(cand)

            if eval_col1.button("Run Benchmark Evaluation"):
                if not gt_boxes:
                    st.info(
                        "ℹ️ No manual ground-truth annotations found. "
                        "To run a benchmark evaluation, draw ground-truth boxes manually on one or more images in Tab 01 (using 'manual' source), "
                        "then run automated detection to compare predictions against your ground truth."
                    )
                else:
                    metrics = evaluate_detections(gt_boxes, pred_boxes, iou_threshold=eval_iou)
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Precision", f"{metrics['precision']:.1%}")
                    m2.metric("Recall", f"{metrics['recall']:.1%}")
                    m3.metric("F1 Score", f"{metrics['f1_score']:.1%}")
                    m4.metric("True Positives", metrics["true_positives"])
                    m5.metric("False Positives", metrics["false_positives"])
                    m6.metric("False Negatives", metrics["false_negatives"])
                    st.caption(
                        f"Evaluated across {len(gt_boxes)} ground-truth box(es) and {len(pred_boxes)} prediction(s) at IoU threshold = {eval_iou:.2f}."
                    )

        # Section 4: Dataset Integrity Validation
        st.markdown("---")
        st.markdown("#### 🔍 Pre-Export Dataset Validation")
        approved_classes = sorted(set(r["class"] for r in all_labels if r.get("class")))
        dataset_classes = approved_classes if approved_classes else get_known_classes()

        val_result = validate_dataset(
            images=st.session_state.images,
            labels=st.session_state.labels,
            classes=dataset_classes,
        )

        if val_result["is_valid"]:
            st.success(
                f"✅ **Dataset Validation Passed:** {val_result['total_images']} image(s) and {val_result['total_boxes']} bounding box(es) "
                "meet all YOLO normalization, dimension, and schema requirements."
            )
        else:
            st.error(
                f"❌ **Dataset Validation Failed ({len(val_result['errors'])} blocking error(s)):** "
                "Export is blocked until the following issues are addressed:"
            )
            for err in val_result["errors"]:
                st.markdown(f"- 🛑 {err}")

        if val_result["warnings"]:
            with st.expander(
                f"⚠️ Validation Warnings ({len(val_result['warnings'])} non-blocking notice(s))",
                expanded=not val_result["is_valid"],
            ):
                for warn in val_result["warnings"]:
                    st.markdown(f"- ⚠️ {warn}")

        # Section 5: Train / Validation Split Configuration
        st.markdown("---")
        st.markdown("#### 🔀 Train / Validation Split Configuration")
        split_col1, split_col2 = st.columns(2)
        with split_col1:
            train_pct = st.slider(
                "Train Split Proportion",
                min_value=50,
                max_value=95,
                value=int(persisted_s.get("train_pct", 80)),
                step=5,
                format="%d%%",
                help="Percentage of images allocated to images/train and labels/train.",
            )
            st.session_state.persisted_settings["train_pct"] = train_pct
        with split_col2:
            seed_val = st.number_input(
                "Deterministic Random Seed",
                min_value=0,
                max_value=999999,
                value=int(persisted_s.get("seed_val", 42)),
                step=1,
                help="Integer seed for reproducible, deterministic train/val partitioning.",
            )
            st.session_state.persisted_settings["seed_val"] = int(seed_val)

        train_names, val_names, split_warnings, train_counts, val_counts = split_dataset(
            image_names=image_names,
            labels=st.session_state.labels,
            train_ratio=train_pct / 100.0,
            seed=int(seed_val),
        )

        for sw in split_warnings:
            st.warning(f"⚠️ {sw}")

        # Section 6: Export Preview
        st.markdown("#### 📋 Export Preview & Class Distribution")
        prev_col1, prev_col2, prev_col3 = st.columns(3)
        train_box_total = sum(train_counts.values())
        val_box_total = sum(val_counts.values())

        prev_col1.metric("Train Split", f"{len(train_names)} images", f"{train_box_total} boxes")
        prev_col2.metric("Validation Split", f"{len(val_names)} images", f"{val_box_total} boxes")
        prev_col3.metric("Total Export", f"{len(image_names)} images", f"{train_box_total + val_box_total} boxes")

        if dataset_classes:
            preview_rows = []
            for idx, cls_name in enumerate(dataset_classes):
                t_cnt = train_counts.get(cls_name, 0)
                v_cnt = val_counts.get(cls_name, 0)
                preview_rows.append(
                    {
                        "Class ID": idx,
                        "Class Name": cls_name,
                        "Train Boxes": t_cnt,
                        "Validation Boxes": v_cnt,
                        "Total Boxes": t_cnt + v_cnt,
                    }
                )
            st.dataframe(preview_rows, use_container_width=True)

            with st.expander("🏷️ YOLO Class ID Index Mapping (`classes.txt` / `dataset.yaml`)", expanded=True):
                st.caption(
                    "Ultralytics YOLO uses the zero-based integer index below in `.txt` label files "
                    "(`<class_id> <x_center> <y_center> <w> <h>`) and `dataset.yaml`."
                )
                map_cols = st.columns(min(4, max(1, len(dataset_classes))))
                for idx, cls_name in enumerate(dataset_classes):
                    with map_cols[idx % len(map_cols)]:
                        st.markdown(f"**`Class ID: {idx}`** $\\rightarrow$ **{cls_name}**")

        # Section 7: YOLO Export Generation & Download
        st.markdown("---")
        st.markdown("#### 📦 Export YOLO Dataset")

        st.caption(
            "Exports standard Ultralytics YOLO layout: `images/train/`, `images/val/`, `labels/train/`, `labels/val/`, `classes.txt`, and escaped `dataset.yaml`. "
            "Images without approved boxes receive empty `.txt` label files to serve as standard background negative samples. "
            "Models are never loaded and weights are never downloaded during validation or export."
        )

        if not val_result["is_valid"]:
            st.button(
                "Generate YOLO Dataset Archive (.zip)",
                type="primary",
                disabled=True,
                help="Export is disabled because pre-export validation found blocking errors.",
            )
            st.caption("🔒 Resolve the validation errors listed above to enable dataset export.")
        else:
            if st.button("Generate YOLO Dataset Archive (.zip)", type="primary"):
                with st.spinner("Compiling YOLO dataset archive (no models loaded)..."):
                    zip_bytes = export_yolo_zip(
                        images=st.session_state.images,
                        labels=st.session_state.labels,
                        train_names=train_names,
                        val_names=val_names,
                        classes=dataset_classes,
                    )
                st.download_button(
                    label="📥 Download YOLO Dataset (.zip)",
                    data=zip_bytes,
                    file_name="labelforge-yolo-dataset.zip",
                    mime="application/zip",
                )
                st.success("🎉 YOLO dataset archive generated successfully!")
