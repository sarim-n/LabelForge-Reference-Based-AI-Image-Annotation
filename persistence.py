"""Local Workspace Persistence module for LabelForge.

Provides atomic, safe, versioned local autosave, restore, and backup export/import
for the LabelForge annotation studio.

Guarantees:
1. Pure JSON + image files (no pickle / unsafe deserialization).
2. Atomic writes using temporary files and filesystem replace.
3. Path traversal protection for safe file and archive handling.
4. Complete isolation from machine learning model objects (PyTorch models,
   processors, and CUDA tensors are never serialized).
5. Comprehensive validation of restored/imported data before applying.
"""

from collections import defaultdict
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import zipfile

from PIL import Image

WORKSPACE_FORMAT = "labelforge-workspace"
WORKSPACE_VERSION = 1
MAX_ZIP_EXTRACT_SIZE = 1024 * 1024 * 1024  # 1 GB safety limit for backup archives
MAX_ZIP_FILE_COUNT = 5000


class WorkspaceError(Exception):
    """Base exception for workspace persistence operations."""
    pass


class WorkspaceValidationError(WorkspaceError):
    """Raised when workspace data, schema, or archive is invalid or corrupted."""
    pass


def sanitize_filename(name: str) -> str:
    """Sanitize a filename to prevent directory traversal and filesystem issues.

    Replaces path separators and illegal characters while preserving a clean basename.
    """
    cleaned = os.path.basename(name).strip()
    # Replace dangerous or reserved characters
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", cleaned)
    # Remove leading dots to prevent hidden files / traversal
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        cleaned = f"file_{int(time.time() * 1000)}"
    return cleaned


def sanitize_class_name(name: str) -> str:
    """Sanitize a class name for use in filesystem folder paths."""
    cleaned = re.sub(r'[\\/*?:"<>|.\s]', "_", name.strip())
    return cleaned if cleaned else "unnamed_class"


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """Atomically write binary data to target_path using a temporary file and replace."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        with open(temp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # On Windows, os.replace can transiently raise PermissionError if a file watcher or Streamlit has the file open
        for attempt in range(5):
            try:
                os.replace(temp_path, target_path)
                break
            except (PermissionError, OSError):
                if attempt == 4:
                    with open(target_path, "wb") as f_out:
                        f_out.write(data)
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                else:
                    time.sleep(0.02 * (attempt + 1))
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise WorkspaceError(f"Failed to atomically write {target_path}: {exc}") from exc


def atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
    """Atomically write a JSON serializable dictionary to target_path."""
    try:
        raw_bytes = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    except Exception as exc:
        raise WorkspaceError(f"Failed to serialize workspace to JSON: {exc}") from exc
    atomic_write_bytes(target_path, raw_bytes)


def is_safe_relpath(path_str: str) -> bool:
    """Verify that a relative path does not escape into parent directories."""
    norm = os.path.normpath(path_str)
    if os.path.isabs(norm) or norm.startswith("..") or norm.startswith(("\\", "/")):
        return False
    # Check drive specifiers on Windows (e.g. C:)
    if len(norm) > 1 and norm[1] == ":":
        return False
    return True


def validate_workspace_schema(data: Any) -> Tuple[bool, Optional[str]]:
    """Validate that the workspace dictionary meets the LabelForge JSON schema.

    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Project file root must be a JSON object."

    fmt = data.get("format")
    if fmt != WORKSPACE_FORMAT:
        return False, f"Incompatible format '{fmt}'. Expected '{WORKSPACE_FORMAT}'."

    ver = data.get("version")
    if ver != WORKSPACE_VERSION:
        return False, f"Unsupported schema version {ver}. Expected version {WORKSPACE_VERSION}."

    if not isinstance(data.get("images"), list):
        return False, "Field 'images' must be a list of image records."

    for idx, img_info in enumerate(data["images"]):
        if not isinstance(img_info, dict):
            return False, f"Image record #{idx} must be a dictionary."
        if not img_info.get("name") or not isinstance(img_info["name"], str):
            return False, f"Image record #{idx} missing valid 'name' string."
        if not img_info.get("path") or not isinstance(img_info["path"], str):
            return False, f"Image record #{idx} missing valid 'path' string."
        if not is_safe_relpath(img_info["path"]):
            return False, f"Image record #{idx} contains unsafe path traversal: '{img_info['path']}'."

    if "labels" in data and not isinstance(data["labels"], dict):
        return False, "Field 'labels' must be a dictionary mapping image names to lists of boxes."

    for img_name, box_list in data.get("labels", {}).items():
        if not isinstance(box_list, list):
            return False, f"Labels for image '{img_name}' must be a list."
        for b_idx, box in enumerate(box_list):
            if not isinstance(box, dict):
                return False, f"Label box #{b_idx} for '{img_name}' must be a dictionary."
            if "class" not in box or not isinstance(box["class"], str):
                return False, f"Label box #{b_idx} for '{img_name}' missing string 'class'."
            coords = box.get("box")
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                return False, f"Label box #{b_idx} for '{img_name}' must have a 4-element 'box' coordinate list."
            if not all(isinstance(c, (int, float)) for c in coords):
                return False, f"Coordinates for label box #{b_idx} in '{img_name}' must be numbers."

    if "pending" in data and not isinstance(data["pending"], dict):
        return False, "Field 'pending' must be a dictionary mapping image names to proposal lists."

    for img_name, prop_list in data.get("pending", {}).items():
        if not isinstance(prop_list, list):
            return False, f"Pending proposals for '{img_name}' must be a list."
        for p_idx, prop in enumerate(prop_list):
            if not isinstance(prop, dict):
                return False, f"Pending proposal #{p_idx} for '{img_name}' must be a dictionary."
            if "class" not in prop or not isinstance(prop["class"], str):
                return False, f"Pending proposal #{p_idx} for '{img_name}' missing string 'class'."
            coords = prop.get("box")
            if not isinstance(coords, (list, tuple)) or len(coords) != 4:
                return False, f"Pending proposal #{p_idx} for '{img_name}' must have 4-element 'box'."

    if "references" in data and not isinstance(data["references"], list):
        return False, "Field 'references' must be a list."

    for r_idx, ref in enumerate(data.get("references", [])):
        if not isinstance(ref, dict):
            return False, f"Reference #{r_idx} must be a dictionary."
        if not ref.get("class") or not isinstance(ref["class"], str):
            return False, f"Reference #{r_idx} missing string 'class'."
        if not ref.get("path") or not isinstance(ref["path"], str):
            return False, f"Reference #{r_idx} missing string 'path'."
        if not is_safe_relpath(ref["path"]):
            return False, f"Reference #{r_idx} contains unsafe path traversal: '{ref['path']}'."

    return True, None


def clean_box_coords(box: Sequence[Any]) -> List[int]:
    """Ensure bounding box coordinates are clean standard Python integers."""
    return [int(round(float(c))) for c in box[:4]]


def build_workspace_manifest(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    refs: Dict[str, List[Image.Image]],
    pending: Dict[str, List[Dict[str, Any]]],
    times: List[Dict[str, Any]],
    rejections: List[int],
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, bytes], Dict[str, bytes]]:
    """Build the JSON manifest and encoded image byte dictionaries for workspace persistence.

    Returns
    -------
    (manifest_dict, image_bytes_map, ref_bytes_map)
        image_bytes_map maps relative paths (e.g. 'images/name.png') to PNG/JPEG bytes.
        ref_bytes_map maps relative paths (e.g. 'references/cls/ref_0.png') to PNG bytes.
    """
    settings_dict = dict(settings or {})

    # Sanitize and encode raw dataset images
    images_manifest: List[Dict[str, Any]] = []
    image_bytes_map: Dict[str, bytes] = {}

    for original_name, img_obj in images.items():
        safe_base = sanitize_filename(original_name)
        # Ensure image has an appropriate extension
        if not re.search(r"\.(jpe?g|png|webp)$", safe_base, re.IGNORECASE):
            safe_base = f"{safe_base}.png"
        rel_path = f"images/{safe_base}"

        img_bytes = getattr(img_obj, "_cached_bytes", None)
        if img_bytes is None:
            buf = io.BytesIO()
            # Save as PNG to preserve full quality without loss
            img_obj.save(buf, format="PNG")
            img_bytes = buf.getvalue()
            try:
                img_obj._cached_bytes = img_bytes
            except Exception:
                pass

        image_bytes_map[rel_path] = img_bytes
        images_manifest.append({
            "name": original_name,
            "path": rel_path,
            "width": int(img_obj.width),
            "height": int(img_obj.height),
        })

    # Sanitize approved labels (keep only serializable primitives)
    clean_labels: Dict[str, List[Dict[str, Any]]] = {}
    for img_name, box_list in labels.items():
        clean_list = []
        for b in box_list:
            clean_list.append({
                "class": str(b.get("class", "object")),
                "box": clean_box_coords(b.get("box", [0, 0, 1, 1])),
                "source": str(b.get("source", "manual")),
                "score": float(b["score"]) if "score" in b and isinstance(b["score"], (int, float)) else None,
                "id": str(b.get("id")) if b.get("id") else None,
            })
        clean_labels[img_name] = clean_list

    # Sanitize pending proposals
    clean_pending: Dict[str, List[Dict[str, Any]]] = {}
    for img_name, prop_list in pending.items():
        clean_list = []
        for p in prop_list:
            clean_list.append({
                "class": str(p.get("class", "object")),
                "box": clean_box_coords(p.get("box", [0, 0, 1, 1])),
                "score": float(p.get("score", 1.0)) if isinstance(p.get("score"), (int, float)) else 1.0,
                "source": str(p.get("source", "auto")),
                "id": str(p.get("id")) if p.get("id") else None,
            })
        clean_pending[img_name] = clean_list

    # Encode visual reference crops
    refs_manifest: List[Dict[str, Any]] = []
    ref_bytes_map: Dict[str, bytes] = {}
    for cls_name, crops in refs.items():
        safe_cls = sanitize_class_name(cls_name)
        for c_idx, crop_img in enumerate(crops):
            rel_path = f"references/{safe_cls}/ref_{c_idx}.png"
            crop_bytes = getattr(crop_img, "_cached_bytes", None)
            if crop_bytes is None:
                buf = io.BytesIO()
                crop_img.save(buf, format="PNG")
                crop_bytes = buf.getvalue()
                try:
                    crop_img._cached_bytes = crop_bytes
                except Exception:
                    pass
            ref_bytes_map[rel_path] = crop_bytes
            refs_manifest.append({
                "class": str(cls_name),
                "path": rel_path,
                "width": int(crop_img.width),
                "height": int(crop_img.height),
            })

    # Sanitize timing records and rejections
    clean_times = [
        {"seconds": float(t.get("seconds", 0.0)), "images": int(t.get("images", 0))}
        for t in times
        if isinstance(t, dict)
    ]
    clean_rejections = [int(r) for r in rejections if isinstance(r, (int, float))]

    manifest = {
        "format": WORKSPACE_FORMAT,
        "version": WORKSPACE_VERSION,
        "saved_at": time.time(),
        "saved_at_iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings_dict,
        "images": images_manifest,
        "labels": clean_labels,
        "pending": clean_pending,
        "references": refs_manifest,
        "times": clean_times,
        "rejections": clean_rejections,
        "metrics_summary": {
            "total_images": len(images),
            "total_approved_boxes": sum(len(v) for v in clean_labels.values()),
            "total_pending_boxes": sum(len(v) for v in clean_pending.values()),
            "total_references": len(refs_manifest),
            "total_rejections": len(clean_rejections),
        },
    }

    return manifest, image_bytes_map, ref_bytes_map


def save_workspace(
    workspace_dir: str | Path,
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    refs: Dict[str, List[Image.Image]],
    pending: Dict[str, List[Dict[str, Any]]],
    times: List[Dict[str, Any]],
    rejections: List[int],
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Autosave the active workspace to local disk using atomic file writes.

    Parameters
    ----------
    workspace_dir : str or Path
        Target root directory for the local workspace (e.g. 'work/workspace').
    images, labels, refs, pending, times, rejections, settings :
        Workspace in-memory state.

    Returns
    -------
    dict
        Metadata about the successful save (timestamp, image counts, etc.).
    """
    ws_path = Path(workspace_dir)
    ws_path.mkdir(parents=True, exist_ok=True)

    manifest, img_bytes_map, ref_bytes_map = build_workspace_manifest(
        images=images,
        labels=labels,
        refs=refs,
        pending=pending,
        times=times,
        rejections=rejections,
        settings=settings,
    )

    # 1. Write image files only if not already present or updated
    for rel_path, raw_bytes in img_bytes_map.items():
        file_path = ws_path / rel_path
        if not file_path.exists() or file_path.stat().st_size != len(raw_bytes):
            atomic_write_bytes(file_path, raw_bytes)

    # 2. Write reference crop files incrementally
    active_ref_rel_paths = set(ref_bytes_map.keys())
    for rel_path, raw_bytes in ref_bytes_map.items():
        file_path = ws_path / rel_path
        if not file_path.exists() or file_path.stat().st_size != len(raw_bytes):
            atomic_write_bytes(file_path, raw_bytes)

    # Clean orphaned reference crop files
    refs_root = ws_path / "references"
    if refs_root.exists():
        for existing in refs_root.glob("*/*.png"):
            rel = f"references/{existing.parent.name}/{existing.name}"
            if rel not in active_ref_rel_paths:
                try:
                    existing.unlink()
                except Exception:
                    pass

    # Clean orphaned image files not in the active manifest
    active_img_rel_paths = set(img_bytes_map.keys())
    images_root = ws_path / "images"
    if images_root.exists():
        for existing in images_root.iterdir():
            if existing.is_file():
                rel = f"images/{existing.name}"
                if rel not in active_img_rel_paths:
                    try:
                        existing.unlink()
                    except Exception:
                        pass

    # 3. Atomically write project.json manifest last (acts as commit point)
    project_json_path = ws_path / "project.json"
    atomic_write_json(project_json_path, manifest)

    return {
        "status": "Saved",
        "time": manifest["saved_at_iso"],
        "timestamp": manifest["saved_at"],
        "images_count": len(images),
        "approved_boxes": manifest["metrics_summary"]["total_approved_boxes"],
        "pending_boxes": manifest["metrics_summary"]["total_pending_boxes"],
        "references_count": manifest["metrics_summary"]["total_references"],
    }


def has_saved_workspace(workspace_dir: str | Path) -> bool:
    """Check if a valid, non-empty workspace project file exists on disk."""
    ws_path = Path(workspace_dir)
    manifest_path = ws_path / "project.json"
    return manifest_path.is_file() and manifest_path.stat().st_size > 10


def load_workspace(workspace_dir: str | Path) -> Dict[str, Any]:
    """Load and validate the local workspace from disk.

    Returns
    -------
    dict
        Decoded workspace dictionary containing:
        'images', 'labels', 'refs', 'pending', 'times', 'rejections', 'settings', 'saved_at'.
    """
    ws_path = Path(workspace_dir)
    manifest_path = ws_path / "project.json"

    if not manifest_path.is_file():
        raise WorkspaceError(f"No workspace project found at '{manifest_path}'.")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        raise WorkspaceValidationError(f"Could not parse workspace project.json: {exc}") from exc

    is_valid, err_msg = validate_workspace_schema(manifest)
    if not is_valid:
        raise WorkspaceValidationError(f"Invalid workspace schema: {err_msg}")

    # Load images safely
    images: Dict[str, Image.Image] = {}
    for img_info in manifest["images"]:
        orig_name = img_info["name"]
        rel_path = img_info["path"]
        img_file = ws_path / rel_path
        if not img_file.is_file():
            raise WorkspaceValidationError(f"Missing image file '{rel_path}' for '{orig_name}'.")
        try:
            with open(img_file, "rb") as f:
                raw_bytes = f.read()
                img_obj = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img_obj.load()  # Ensure image data is loaded into memory
                img_obj._cached_bytes = raw_bytes
            images[orig_name] = img_obj
        except Exception as exc:
            raise WorkspaceValidationError(f"Failed to decode image '{rel_path}': {exc}") from exc

    # Load labels and pending
    labels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for img_name, box_list in manifest.get("labels", {}).items():
        if img_name in images:
            labels[img_name] = box_list

    pending: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for img_name, prop_list in manifest.get("pending", {}).items():
        if img_name in images:
            pending[img_name] = prop_list

    # Load visual references
    refs: Dict[str, List[Image.Image]] = defaultdict(list)
    for ref_info in manifest.get("references", []):
        cls_name = ref_info["class"]
        rel_path = ref_info["path"]
        ref_file = ws_path / rel_path
        if ref_file.is_file():
            try:
                with open(ref_file, "rb") as f:
                    raw_crop_bytes = f.read()
                    crop_obj = Image.open(io.BytesIO(raw_crop_bytes)).convert("RGB")
                    crop_obj.load()
                    crop_obj._cached_bytes = raw_crop_bytes
                refs[cls_name].append(crop_obj)
            except Exception:
                pass  # Skip corrupted individual crop gracefully

    times: List[Dict[str, Any]] = manifest.get("times", [])
    rejections: List[int] = manifest.get("rejections", [])
    settings: Dict[str, Any] = manifest.get("settings", {})

    return {
        "images": images,
        "labels": labels,
        "refs": refs,
        "pending": pending,
        "times": times,
        "rejections": rejections,
        "settings": settings,
        "saved_at": manifest.get("saved_at_iso") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": manifest.get("saved_at", time.time()),
    }


def export_backup_zip(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    refs: Dict[str, List[Image.Image]],
    pending: Dict[str, List[Dict[str, Any]]],
    times: List[Dict[str, Any]],
    rejections: List[int],
    settings: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Generate a portable, self-contained project backup archive (.zip).

    Contains project.json, all images, and visual reference crops.
    Completely independent from the YOLO dataset export.
    """
    manifest, img_bytes_map, ref_bytes_map = build_workspace_manifest(
        images=images,
        labels=labels,
        refs=refs,
        pending=pending,
        times=times,
        rejections=rejections,
        settings=settings,
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. project.json manifest
        manifest_raw = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        zf.writestr("project.json", manifest_raw)

        # 2. Raw dataset images
        for rel_path, raw_bytes in img_bytes_map.items():
            zf.writestr(rel_path, raw_bytes)

        # 3. Visual reference crops
        for rel_path, raw_bytes in ref_bytes_map.items():
            zf.writestr(rel_path, raw_bytes)

    return zip_buf.getvalue()


def restore_backup_zip(zip_data: bytes | io.BytesIO) -> Dict[str, Any]:
    """Validate and restore a project backup archive from bytes.

    Guarantees:
    - Protects against zip-slip directory traversal vulnerabilities.
    - Validates uncompressed size limits and file counts.
    - Validates schema and decodes images strictly with PIL.
    - Never touches the disk or session state if validation fails.

    Returns
    -------
    dict
        Decoded workspace dictionary (same structure as `load_workspace`).
    """
    if isinstance(zip_data, bytes):
        zip_buf = io.BytesIO(zip_data)
    else:
        zip_buf = zip_data

    if not zipfile.is_zipfile(zip_buf):
        raise WorkspaceValidationError("Uploaded file is not a valid ZIP archive.")

    zip_buf.seek(0)
    with zipfile.ZipFile(zip_buf, "r") as zf:
        infolist = zf.infolist()
        if len(infolist) > MAX_ZIP_FILE_COUNT:
            raise WorkspaceValidationError(f"Archive exceeds maximum file count ({MAX_ZIP_FILE_COUNT}).")

        total_uncompressed = sum(info.file_size for info in infolist)
        if total_uncompressed > MAX_ZIP_EXTRACT_SIZE:
            raise WorkspaceValidationError("Archive exceeds maximum uncompressed size limit (1 GB).")

        # Security check: verify no member has path traversal components
        for info in infolist:
            name = info.filename
            if not is_safe_relpath(name):
                raise WorkspaceValidationError(f"Unsafe path in archive: '{name}'.")

        # Check for project.json
        if "project.json" not in zf.namelist():
            raise WorkspaceValidationError("Missing 'project.json' manifest in backup archive.")

        try:
            manifest_raw = zf.read("project.json").decode("utf-8")
            manifest = json.loads(manifest_raw)
        except Exception as exc:
            raise WorkspaceValidationError(f"Could not parse project.json in archive: {exc}") from exc

        is_valid, err_msg = validate_workspace_schema(manifest)
        if not is_valid:
            raise WorkspaceValidationError(f"Invalid backup archive: {err_msg}")

        # Safely extract images
        images: Dict[str, Image.Image] = {}
        for img_info in manifest["images"]:
            orig_name = img_info["name"]
            rel_path = img_info["path"]
            if rel_path not in zf.namelist():
                raise WorkspaceValidationError(f"Archive is missing referenced image '{rel_path}'.")
            try:
                raw_bytes = zf.read(rel_path)
                img_obj = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                img_obj.load()
                images[orig_name] = img_obj
            except Exception as exc:
                raise WorkspaceValidationError(f"Failed to decode image '{rel_path}': {exc}") from exc

        # Extract labels and pending
        labels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for img_name, box_list in manifest.get("labels", {}).items():
            if img_name in images:
                labels[img_name] = box_list

        pending: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for img_name, prop_list in manifest.get("pending", {}).items():
            if img_name in images:
                pending[img_name] = prop_list

        # Extract references
        refs: Dict[str, List[Image.Image]] = defaultdict(list)
        for ref_info in manifest.get("references", []):
            cls_name = ref_info["class"]
            rel_path = ref_info["path"]
            if rel_path in zf.namelist():
                try:
                    crop_bytes = zf.read(rel_path)
                    crop_obj = Image.open(io.BytesIO(crop_bytes)).convert("RGB")
                    crop_obj.load()
                    refs[cls_name].append(crop_obj)
                except Exception:
                    pass

    return {
        "images": images,
        "labels": labels,
        "refs": refs,
        "pending": pending,
        "times": manifest.get("times", []),
        "rejections": manifest.get("rejections", []),
        "settings": manifest.get("settings", {}),
        "saved_at": manifest.get("saved_at_iso") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": manifest.get("saved_at", time.time()),
    }


def clear_saved_workspace(workspace_dir: str | Path) -> bool:
    """Clear the local saved workspace directory on disk.

    Does NOT affect in-memory session data unless explicitly synchronized by the caller.

    Returns
    -------
    bool
        True if a workspace existed and was removed; False otherwise.
    """
    ws_path = Path(workspace_dir)
    if not ws_path.exists():
        return False

    project_json = ws_path / "project.json"
    existed = project_json.exists()

    try:
        shutil.rmtree(ws_path)
        ws_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise WorkspaceError(f"Failed to clear saved workspace directory '{ws_path}': {exc}") from exc

    return existed
