"""Vision and detection module for LabelForge.

Provides image crop utilities, IoU computation, YOLO annotation formatting,
and automated detection interfaces for both zero-shot text-prompted detection
(Grounding DINO Tiny) and prototype visual reference matching (SAM proposals +
frozen DINOv2 feature matching).

Models are strictly lazily loaded so that importing this module does not load
heavy neural networks or download checkpoints on startup. Model weights are
strictly frozen during inference.
"""

from collections import defaultdict
import io
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

from PIL import Image, ImageDraw


def crop_box(image: Image.Image, box: Sequence[int | float]) -> Image.Image:
    """Crop an image using [x1, y1, x2, y2] coordinates clamped to image dimensions."""
    w, h = image.size
    x1 = max(0, min(w, int(round(box[0]))))
    y1 = max(0, min(h, int(round(box[1]))))
    x2 = max(0, min(w, int(round(box[2]))))
    y2 = max(0, min(h, int(round(box[3]))))
    if x2 <= x1 or y2 <= y1:
        x2 = min(w, x1 + 1)
        y2 = min(h, y1 + 1)
    return image.crop((x1, y1, x2, y2)).convert("RGB")


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def box_overlap_containment(box_a: Sequence[float], box_b: Sequence[float]) -> Tuple[float, float]:
    """Calculate both IoU and Containment (Intersection over Min-Area) between two bounding boxes.

    Boxes are in [x1, y1, x2, y2] format.
    Returns:
        tuple of (iou, containment)
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    min_area = min(area_a, area_b)
    containment = intersection / min_area if min_area > 0 else 0.0

    return iou, containment


def yolo_line(class_id: int, box: Sequence[float], width: int, height: int) -> str:
    """Convert an [x1, y1, x2, y2] pixel box to standard YOLO normalized line format.

    Format: <class_id> <x_center> <y_center> <width> <height>
    Normalized to [0.0, 1.0] relative to image dimensions.
    """
    x1, y1, x2, y2 = box
    box_w = max(0.0, float(x2 - x1))
    box_h = max(0.0, float(y2 - y1))
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    norm_x = x_center / max(1, width)
    norm_y = y_center / max(1, height)
    norm_w = box_w / max(1, width)
    norm_h = box_h / max(1, height)

    return f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def draw_box_with_label(
    draw_ctx: ImageDraw.ImageDraw,
    box: Sequence[int | float],
    label_text: str,
    color: str = "#2b7a5a",
    text_color: str = "#ffffff",
    line_width: int = 3,
) -> None:
    """Draw a bounding box with an attached badge tag displaying the class ID and name."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    draw_ctx.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    if label_text:
        try:
            bbox = draw_ctx.textbbox((x1, y1), label_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label_text) * 8, 14
        pad = 2
        badge_y1 = max(0, y1 - th - 2 * pad)
        badge_y2 = badge_y1 + th + 2 * pad
        badge_x2 = min(draw_ctx.im.size[0], x1 + tw + 2 * pad)
        draw_ctx.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color)
        draw_ctx.text((x1 + pad, badge_y1 + pad), label_text, fill=text_color)


def parse_canvas_objects(
    objects: Sequence[Dict[str, Any]],
    canvas_w: int,
    canvas_h: int,
    img_w: int,
    img_h: int,
) -> tuple[Dict[int, List[int]], Dict[int, List[int]], List[Dict[str, Any]]]:
    """Parse Fabric.js canvas objects back into image pixel coordinates.

    Returns:
        tuple of:
        - approved_updates: {index -> [x1, y1, x2, y2]}
        - proposal_updates: {index -> [x1, y1, x2, y2]}
        - new_drawn_boxes: list of {"box": [x1, y1, x2, y2], "label": str}
    """
    scale_x = img_w / max(1, canvas_w)
    scale_y = img_h / max(1, canvas_h)

    approved_updates: Dict[int, List[int]] = {}
    proposal_updates: Dict[int, List[int]] = {}
    new_drawn_boxes: List[Dict[str, Any]] = []

    for obj in objects:
        if obj.get("type") not in ("LabeledRect", "rect"):
            continue

        sx = float(obj.get("scaleX", 1.0))
        sy = float(obj.get("scaleY", 1.0))
        w = float(obj.get("width", 0.0)) * sx
        h = float(obj.get("height", 0.0)) * sy
        ox = obj.get("originX", "left")
        oy = obj.get("originY", "top")
        off_x = 0.5 if ox == "center" else (1.0 if ox == "right" else 0.0)
        off_y = 0.5 if oy == "center" else (1.0 if oy == "bottom" else 0.0)
        left = float(obj.get("left", 0.0)) - off_x * w
        top = float(obj.get("top", 0.0)) - off_y * h

        x1 = max(0, min(img_w, int(round(left * scale_x))))
        y1 = max(0, min(img_h, int(round(top * scale_y))))
        x2 = max(0, min(img_w, int(round((left + w) * scale_x))))
        y2 = max(0, min(img_h, int(round((top + h) * scale_y))))

        if x2 <= x1 or y2 <= y1:
            continue

        box_coords = [x1, y1, x2, y2]
        label = str(obj.get("label", ""))

        def _extract_cls(lbl: str) -> str:
            if not lbl:
                return ""
            txt = str(lbl).strip()
            if txt.startswith("[ID ") and "]" in txt:
                txt = txt.split("]", 1)[1].strip()
            return txt

        if label.startswith("Proposal #"):
            try:
                num_str = label.split(":")[0].replace("Proposal #", "").strip()
                p_idx = int(num_str) - 1
                proposal_updates[p_idx] = box_coords
            except Exception:
                new_drawn_boxes.append({"box": box_coords, "label": label, "class": _extract_cls(label)})
        elif label.startswith("Approved #"):
            try:
                num_str = label.split(":")[0].replace("Approved #", "").strip()
                a_idx = int(num_str) - 1
                approved_updates[a_idx] = box_coords
            except Exception:
                new_drawn_boxes.append({"box": box_coords, "label": label, "class": _extract_cls(label)})
        else:
            new_drawn_boxes.append({"box": box_coords, "label": label, "class": _extract_cls(label)})

    return approved_updates, proposal_updates, new_drawn_boxes


def validate_dataset(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    classes: Sequence[str],
) -> Dict[str, Any]:
    """Validate images and annotations prior to YOLO export.

    Checks:
    - Missing or unreadable image objects
    - Duplicate image filenames and stem collisions
    - Out-of-bounds bounding boxes
    - Zero-area or inverted bounding boxes
    - Malformed label dictionaries
    - Unknown class references
    """
    errors: List[str] = []
    warnings: List[str] = []
    class_set = set(classes)
    stem_map: Dict[str, str] = {}
    total_boxes = 0
    class_counts: Dict[str, int] = defaultdict(int)
    empty_images_count = 0

    if not images:
        errors.append("Dataset is empty. No images uploaded to export.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_images": 0,
            "total_boxes": 0,
            "empty_images": 0,
            "class_counts": {},
        }

    for name, img in images.items():
        # 1. Unreadable image check
        if img is None or not hasattr(img, "size") or img.width <= 0 or img.height <= 0:
            errors.append(f"Image '{name}' is missing or has invalid dimensions.")
            continue

        w, h = img.size

        # 2. Duplicate stem collision check
        stem = name.rsplit(".", 1)[0].lower()
        if stem in stem_map and stem_map[stem] != name:
            errors.append(
                f"Filename collision: '{name}' and '{stem_map[stem]}' map to the same label file stem '{stem}.txt'."
            )
        else:
            stem_map[stem] = name

        img_labels = labels.get(name, [])
        if not img_labels:
            empty_images_count += 1
            continue

        for idx, entry in enumerate(img_labels):
            total_boxes += 1
            if not isinstance(entry, dict):
                errors.append(f"Image '{name}' annotation #{idx + 1} is malformed (not a dictionary).")
                continue

            # Class check
            cls_name = entry.get("class")
            if not cls_name or not str(cls_name).strip():
                errors.append(f"Image '{name}' box #{idx + 1} is missing a class label.")
            elif cls_name not in class_set:
                errors.append(f"Image '{name}' box #{idx + 1} references undefined class '{cls_name}'.")
            else:
                class_counts[cls_name] += 1

            # Box coordinates check
            box = entry.get("box")
            if not box or len(box) != 4:
                errors.append(f"Image '{name}' box #{idx + 1} coordinates are malformed: {box}.")
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in box]
            except (ValueError, TypeError):
                errors.append(f"Image '{name}' box #{idx + 1} contains non-numeric coordinates: {box}.")
                continue

            # Zero-area or inverted check
            if x2 <= x1 or y2 <= y1:
                errors.append(
                    f"Image '{name}' box #{idx + 1} has zero or negative area: [{x1}, {y1}, {x2}, {y2}]."
                )

            # Out of bounds check
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                warnings.append(
                    f"Image '{name}' box #{idx + 1} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}] exceeds image bounds ({w}x{h}). Coordinates will be clamped to image boundaries during export."
                )

    # Informational warnings
    if empty_images_count > 0:
        warnings.append(
            f"{empty_images_count} image(s) have no approved bounding boxes. Empty label text files will be generated as standard YOLO background samples."
        )

    for c in classes:
        cnt = class_counts.get(c, 0)
        if cnt == 0:
            warnings.append(f"Class '{c}' is defined in dataset classes but has 0 approved bounding boxes.")
        elif cnt < 3:
            warnings.append(f"Class '{c}' has only {cnt} approved example(s). Recommended minimum is at least 3-5 examples for training.")

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "total_images": len(images),
        "total_boxes": total_boxes,
        "empty_images": empty_images_count,
        "class_counts": dict(class_counts),
    }


def split_dataset(
    image_names: Sequence[str],
    labels: Dict[str, List[Dict[str, Any]]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], Dict[str, int], Dict[str, int]]:
    """Deterministically partition images into train and validation sets.

    Guarantees:
    - Train and val sets are strictly disjoint.
    - Deterministic and reproducible with an integer seed.
    - Warns if dataset is too small or if classes are absent from split.
    """
    import random

    warnings: List[str] = []
    shuffled_names = list(image_names)
    rng = random.Random(seed)
    rng.shuffle(shuffled_names)

    total = len(shuffled_names)
    if total <= 1:
        warnings.append("Dataset has only 1 image. Split cannot create non-empty train and validation sets.")
        return shuffled_names, [], warnings, {}, {}

    train_count = int(round(total * train_ratio))
    # Ensure at least 1 image in each set when total >= 2
    train_count = max(1, min(total - 1, train_count))

    train_names = sorted(shuffled_names[:train_count])
    val_names = sorted(shuffled_names[train_count:])

    # Track box counts per split
    train_class_counts: Dict[str, int] = defaultdict(int)
    val_class_counts: Dict[str, int] = defaultdict(int)

    for n in train_names:
        for entry in labels.get(n, []):
            if "class" in entry:
                train_class_counts[entry["class"]] += 1

    for n in val_names:
        for entry in labels.get(n, []):
            if "class" in entry:
                val_class_counts[entry["class"]] += 1

    # Check for class representation in both splits
    all_seen_classes = set(train_class_counts.keys()) | set(val_class_counts.keys())
    for c in all_seen_classes:
        t_cnt = train_class_counts.get(c, 0)
        v_cnt = val_class_counts.get(c, 0)
        if t_cnt == 0:
            warnings.append(f"Class '{c}' has 0 annotations in the train split (all {v_cnt} are in validation).")
        elif v_cnt == 0:
            warnings.append(f"Class '{c}' has 0 annotations in the validation split (all {t_cnt} are in training).")

    return train_names, val_names, warnings, dict(train_class_counts), dict(val_class_counts)


def export_yolo_zip(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    train_names: Sequence[str],
    val_names: Sequence[str],
    classes: Sequence[str],
) -> bytes:
    """Export complete normalized YOLO dataset zip archive.

    Structure:
    images/train/<stem>.jpg
    images/val/<stem>.jpg
    labels/train/<stem>.txt
    labels/val/<stem>.txt
    classes.txt
    dataset.yaml
    """
    import io
    import zipfile

    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Process train split
        for name in train_names:
            img = images[name]
            stem = name.rsplit(".", 1)[0]
            img_buf = io.BytesIO()
            img.save(img_buf, format="JPEG")
            zf.writestr(f"images/train/{stem}.jpg", img_buf.getvalue())

            lines = []
            for entry in labels.get(name, []):
                cls_name = entry.get("class")
                if cls_name in class_to_id:
                    cid = class_to_id[cls_name]
                    # Clamp coordinates to ensure valid YOLO format
                    raw_b = entry["box"]
                    w, h = img.size
                    x1 = max(0, min(w, int(round(raw_b[0]))))
                    y1 = max(0, min(h, int(round(raw_b[1]))))
                    x2 = max(0, min(w, int(round(raw_b[2]))))
                    y2 = max(0, min(h, int(round(raw_b[3]))))
                    if x2 > x1 and y2 > y1:
                        lines.append(yolo_line(cid, [x1, y1, x2, y2], w, h))
            # Empty text file for negative background images
            label_text = "\n".join(lines) + ("\n" if lines else "")
            zf.writestr(f"labels/train/{stem}.txt", label_text)

        # 2. Process val split
        for name in val_names:
            img = images[name]
            stem = name.rsplit(".", 1)[0]
            img_buf = io.BytesIO()
            img.save(img_buf, format="JPEG")
            zf.writestr(f"images/val/{stem}.jpg", img_buf.getvalue())

            lines = []
            for entry in labels.get(name, []):
                cls_name = entry.get("class")
                if cls_name in class_to_id:
                    cid = class_to_id[cls_name]
                    raw_b = entry["box"]
                    w, h = img.size
                    x1 = max(0, min(w, int(round(raw_b[0]))))
                    y1 = max(0, min(h, int(round(raw_b[1]))))
                    x2 = max(0, min(w, int(round(raw_b[2]))))
                    y2 = max(0, min(h, int(round(raw_b[3]))))
                    if x2 > x1 and y2 > y1:
                        lines.append(yolo_line(cid, [x1, y1, x2, y2], w, h))
            label_text = "\n".join(lines) + ("\n" if lines else "")
            zf.writestr(f"labels/val/{stem}.txt", label_text)

        # 3. classes.txt
        zf.writestr("classes.txt", "\n".join(classes) + "\n")

        # 4. dataset.yaml with proper string escaping
        yaml_lines = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        for idx, cls_name in enumerate(classes):
            escaped_name = cls_name.replace("'", "''")
            yaml_lines.append(f"  {idx}: '{escaped_name}'")
        yaml_lines.append("")
        zf.writestr("dataset.yaml", "\n".join(yaml_lines))

    return zip_buffer.getvalue()


def evaluate_detections(
    ground_truth_boxes: Sequence[Dict[str, Any]],
    predicted_boxes: Sequence[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate Precision, Recall, and F1 score against reference ground-truth annotations.

    Matches predictions to ground-truth boxes of the same class using greedy IoU assignment.
    """
    gt_matched = [False] * len(ground_truth_boxes)
    tp = 0

    sorted_indices = sorted(
        range(len(predicted_boxes)),
        key=lambda i: predicted_boxes[i].get("score", 1.0),
        reverse=True,
    )

    for p_idx in sorted_indices:
        pred = predicted_boxes[p_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(ground_truth_boxes):
            if not gt_matched[g_idx] and pred["class"] == gt["class"]:
                overlap = box_iou(pred["box"], gt["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_idx = g_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
            tp += 1

    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "iou_threshold": iou_threshold,
    }


def extract_class_list(prompt: str | Sequence[str]) -> List[str]:
    """Extract individual clean class names from string or sequence input.

    Splits on commas, semicolons, newlines, slashes, ampersands, and 'and'/'or' conjunctions.
    Preserves multi-word class names like 'coffee cup' or 'dining table'.
    Strips leading articles ('a ', 'an ', 'the ').
    """
    if isinstance(prompt, (list, tuple, set)):
        items = []
        for p in prompt:
            items.extend(extract_class_list(str(p)))
        return items

    text = str(prompt).strip()
    if not text:
        return []

    # Replace conjunctions surrounded by whitespace (' and ', ' or ', ' & ') with comma
    text = re.sub(r"\s+(?:and|or|&)\s+", ", ", text, flags=re.IGNORECASE)
    # Split on commas, semicolons, slashes, pipes, newlines, and periods
    parts = re.split(r"[,;/\n|\.]+", text)

    cleaned = []
    for p in parts:
        item = p.strip()
        # Strip leading articles ('a ', 'an ', 'the ')
        item = re.sub(r"^(?:a|an|the)\s+", "", item, flags=re.IGNORECASE).strip()
        if item:
            cleaned.append(item)
    return cleaned


def format_grounding_dino_prompt(prompt: str | Sequence[str]) -> Tuple[str, List[str]]:
    """Format user class names into Grounding DINO text prompt format.

    Extracts separate class names even when entered as 'pillow and bed', 'pillow, bed', etc.
    Grounding DINO expects lowercased phrases separated by periods and ending with a period,
    e.g. 'pillow . bed .'.

    Returns
    -------
    tuple of (formatted_text_prompt, list_of_cleaned_original_class_names)
    """
    raw_items = extract_class_list(prompt)
    if not raw_items:
        return "", []

    unique_lowered: List[str] = []
    unique_original: List[str] = []
    for orig in raw_items:
        low = orig.lower()
        if low not in unique_lowered:
            unique_lowered.append(low)
            unique_original.append(orig)

    formatted_text = " . ".join(unique_lowered) + " ."
    return formatted_text, unique_original


def match_label_to_class(predicted_label: str, class_list: Sequence[str]) -> str:
    """Match a predicted label string back to the user's defined class names."""
    clean_label = predicted_label.strip().rstrip(".").strip()
    if not clean_label:
        return class_list[0] if class_list else "object"

    clean_lower = clean_label.lower()

    # 1. Exact match
    for orig in class_list:
        if orig.lower() == clean_lower:
            return orig

    # 2. Token word match for merged labels (e.g. 'pillow bed' -> 'pillow' or 'bed')
    words = clean_lower.split()
    for w in words:
        for orig in class_list:
            if orig.lower() == w:
                return orig

    # 3. Substring match fallback (e.g. 'a bottle' -> 'bottle')
    for orig in class_list:
        if orig.lower() in clean_lower or clean_lower in orig.lower():
            return orig

    return clean_label


class GroundingDinoDetector:
    """Pretrained Grounding DINO Tiny detector for zero-shot text-prompted detection.

    Model weights remain strictly frozen. Imports of transformers and torch,
    as well as model loading, are deferred until the first prediction call.
    Checkpoints are loaded lazily on demand.
    """

    MODEL_ID = "IDEA-Research/grounding-dino-tiny"

    def __init__(self, device: str = "auto"):
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO Tiny requires optional model dependencies ('torch' and 'transformers'). "
                "Activate your virtual environment and run: pip install -r requirements-model.txt"
            ) from exc

        from config import resolve_device

        self.device_str, self.device_status = resolve_device(device, torch)
        if self.device_str is None:
            raise RuntimeError(self.device_status)

        self.torch = torch
        self.device = torch.device(self.device_str)

        try:
            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.MODEL_ID)
            self.model.to(self.device)
            self.model.eval()

            # Strictly freeze all parameters (zero-shot inference only)
            for param in self.model.parameters():
                param.requires_grad = False

            import inspect
            sig_params = inspect.signature(self.processor.post_process_grounded_object_detection).parameters
            self._box_thresh_param = "box_threshold" if "box_threshold" in sig_params else "threshold"

            # Optimize CPU thread allocation to prevent thread thrashing
            if self.device_str == "cpu":
                try:
                    import os
                    # Allocate optimal threads for CPU inference
                    num_threads = max(1, min(8, os.cpu_count() or 4))
                    torch.set_num_threads(num_threads)
                except Exception:
                    pass
        except Exception as exc:
            err_msg = str(exc)
            if "out of memory" in err_msg.lower():
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"CUDA out of memory while loading Grounding DINO Tiny: {exc}. "
                    "Switch compute device to 'CPU' in the sidebar to run on system memory."
                ) from exc
            raise RuntimeError(
                f"Failed to load Grounding DINO Tiny ({self.MODEL_ID}): {exc}. "
                "Verify internet connectivity for checkpoint download or check PyTorch compatibility."
            ) from exc

    def predict(
        self,
        image: Image.Image,
        prompt: str | Sequence[str],
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        target_size: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Detect objects in an image matching text prompts without modifying model weights.

        Correctly separates and identifies multiple classes (e.g. 'pillow and bed') into
        distinct class labels.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to scan.
        prompt : str or list of str
            Class names or text description, e.g. 'pillow and bed' or ['pillow', 'bed'].
        box_threshold : float
            Confidence threshold for proposal bounding boxes (similarity score, not a probability).
        text_threshold : float
            Threshold for text-to-box alignment.
        target_size : int, optional
            Maximum edge size for inference scaling (e.g. 640 or 800).
            Significantly accelerates CPU inference while accurately preserving original image box coordinates.

        Returns
        -------
        list of dicts
            Each dict has keys: 'class', 'box' [x1, y1, x2, y2], 'score', 'source'.
        """
        formatted_text, class_names = format_grounding_dino_prompt(prompt)
        if not formatted_text or not class_names:
            return []

        w, h = image.size
        rgb_img = image.convert("RGB")

        # Default target_size for CPU to prevent 30s+ inference times
        if target_size is None and self.device_str == "cpu":
            target_size = 640

        # Scale image down if target_size is set and image exceeds it
        if target_size and max(w, h) > target_size:
            scale_ratio = target_size / max(w, h)
            new_w = max(1, int(round(w * scale_ratio)))
            new_h = max(1, int(round(h * scale_ratio)))
            scaled_img = rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            scaled_img = rgb_img

        try:
            processor_kwargs: Dict[str, Any] = {
                "images": scaled_img,
                "text": formatted_text,
                "return_tensors": "pt",
            }
            if target_size:
                processor_kwargs["size"] = {
                    "shortest_edge": min(600, int(round(target_size * 0.75))),
                    "longest_edge": target_size,
                }

            inputs = self.processor(**processor_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.inference_mode():
                outputs = self.model(**inputs)

            # Map token spans for each class
            input_ids = inputs["input_ids"][0].tolist()
            dot_id = self.processor.tokenizer.convert_tokens_to_ids(".")
            sep_id = self.processor.tokenizer.convert_tokens_to_ids("[SEP]")
            cls_id = self.processor.tokenizer.convert_tokens_to_ids("[CLS]")

            class_token_spans: List[List[int]] = []
            curr_span: List[int] = []
            for idx, tid in enumerate(input_ids):
                if tid in (cls_id, sep_id):
                    continue
                if tid == dot_id:
                    if curr_span:
                        class_token_spans.append(curr_span)
                        curr_span = []
                else:
                    curr_span.append(idx)

            detections: List[Dict[str, Any]] = []

            # If token spans match class count, perform precise per-class probability assignment
            if len(class_token_spans) == len(class_names):
                logits = outputs.logits[0]  # shape: (num_queries, 256)
                probs = self.torch.sigmoid(logits)
                pred_boxes = outputs.pred_boxes[0]  # shape: (num_queries, 4) [cx, cy, bw, bh]

                class_scores = []
                for span in class_token_spans:
                    span_tensor = self.torch.tensor(span, device=probs.device)
                    # Maximum cross-attention probability across the tokens of this class
                    span_prob = probs[:, span_tensor].max(dim=-1)[0]
                    class_scores.append(span_prob)

                # Stack to shape (num_queries, num_classes)
                class_scores_tensor = self.torch.stack(class_scores, dim=-1)
                best_scores, best_class_indices = class_scores_tensor.max(dim=-1)

                keep_mask = (best_scores >= box_threshold) & (best_scores >= text_threshold)
                keep_indices = keep_mask.nonzero(as_tuple=True)[0]

                raw_dets = []
                for q_idx in keep_indices:
                    c_idx = best_class_indices[q_idx].item()
                    c_name = class_names[c_idx]
                    score_val = float(best_scores[q_idx].item())

                    cx, cy, bw, bh = pred_boxes[q_idx].tolist()
                    xmin = (cx - bw / 2.0) * w
                    ymin = (cy - bh / 2.0) * h
                    xmax = (cx + bw / 2.0) * w
                    ymax = (cy + bh / 2.0) * h

                    x1 = max(0, min(w, int(round(xmin))))
                    y1 = max(0, min(h, int(round(ymin))))
                    x2 = max(0, min(w, int(round(xmax))))
                    y2 = max(0, min(h, int(round(ymax))))

                    if x2 > x1 and y2 > y1:
                        raw_dets.append(
                            {
                                "class": c_name,
                                "box": [x1, y1, x2, y2],
                                "score": round(score_val, 4),
                                "source": "auto",
                            }
                        )

                # Apply per-class Non-Maximum Suppression (NMS) to eliminate duplicate boxes of the same class
                by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for d in raw_dets:
                    by_class[d["class"]].append(d)

                for cls_name, items in by_class.items():
                    items.sort(key=lambda x: x["score"], reverse=True)
                    cls_kept: List[Dict[str, Any]] = []
                    for cand in items:
                        if not any(box_iou(cand["box"], k["box"]) > 0.50 for k in cls_kept):
                            cls_kept.append(cand)
                    detections.extend(cls_kept)

            else:
                # Fallback to standard post-processor if token spans couldn't be resolved
                box_param_name = getattr(self, "_box_thresh_param", "threshold")
                post_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "text_threshold": text_threshold,
                    "target_sizes": [(h, w)],
                    box_param_name: box_threshold,
                }
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    **post_kwargs,
                )
                if results:
                    detection_result = results[0]
                    boxes_tensor = detection_result.get("boxes", [])
                    scores_tensor = detection_result.get("scores", [])
                    raw_labels = detection_result.get("text_labels", detection_result.get("labels", []))

                    for box_t, score_t, label_raw in zip(boxes_tensor, scores_tensor, raw_labels):
                        box_coords = box_t.tolist()
                        xmin, ymin, xmax, ymax = box_coords
                        x1 = max(0, min(w, int(round(xmin))))
                        y1 = max(0, min(h, int(round(ymin))))
                        x2 = max(0, min(w, int(round(xmax))))
                        y2 = max(0, min(h, int(round(ymax))))

                        if x2 > x1 and y2 > y1:
                            assigned_class = match_label_to_class(str(label_raw), class_names)
                            score_val = float(score_t.item()) if hasattr(score_t, "item") else float(score_t)
                            detections.append(
                                {
                                    "class": assigned_class,
                                    "box": [x1, y1, x2, y2],
                                    "score": round(score_val, 4),
                                    "source": "auto",
                                }
                            )

        except (self.torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" in str(exc).lower():
                if self.device_str == "cuda" and hasattr(self.torch.cuda, "empty_cache"):
                    self.torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory while executing Grounding DINO Tiny inference. "
                    "Switch Compute Device to 'CPU' in the sidebar or reduce input image resolution."
                ) from exc
            raise

        finally:
            # Low-memory clean-up for ~4 GB GPUs
            if self.device_str == "cuda" and hasattr(self.torch.cuda, "empty_cache"):
                self.torch.cuda.empty_cache()

        return sorted(detections, key=lambda d: d["score"], reverse=True)



def generate_opencv_proposals(
    image: Image.Image,
    limit: int = 240,
    min_size: int = 16,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.95,
    iou_thresh: float = 0.82,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """Generate class-neutral candidate object bounding boxes using OpenCV image processing.

    This generator proposes possible regions of interest WITHOUT assigning semantic
    class labels. Semantic matching is performed downstream by frozen DINOv2 visual
    similarity against user-approved reference crops.

    Techniques used:
    1. Selective Search: If OpenCV ximgproc module (cv2.ximgproc.segmentation) is available,
       it is utilized directly.
    2. Documented Fallback: When cv2.ximgproc is unavailable (e.g. standard opencv-python),
       a multi-cue contour + region proposal system is used:
       a) MSER (Maximally Stable Extremal Regions) for stable intensity blobs.
       b) Multi-scale Canny edge detection with directional morphological closing
          (bridging object/vehicle boundaries across multiple kernels).
       c) Adaptive thresholding with morphological closure to capture contrast outlines.
       d) Multi-scale, multi-aspect spatial anchor proposals covering the visual field.
    3. Coordinate clipping: All proposals are strictly clipped to image boundaries [0, W] and [0, H].
    4. Size and aspect filtering: Discard degenerate (x2 <= x1, y2 <= y1), tiny (< min_size px),
       and extreme area ratio (< min_area_ratio or > max_area_ratio) boxes.
    5. Duplicate & overlap suppression: Sorts candidate regions by saliency score descending
       and applies IoU non-maximum suppression (threshold `iou_thresh`).
    6. Returns up to `limit` deduplicated proposal tuples: ((x1, y1, x2, y2), candidate_score).
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV candidate region generation requires opencv-python and numpy. "
            "Install with: pip install opencv-python numpy"
        ) from exc

    rgb_image = image.convert("RGB")
    orig_w, orig_h = rgb_image.size
    img_area = float(max(1, orig_w * orig_h))
    cv_img = np.array(rgb_image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    raw_candidates: List[Tuple[int, int, int, int, float]] = []

    # 1. Selective Search if cv2.ximgproc is available
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "segmentation"):
        try:
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(cv_img)
            ss.switchToSelectiveSearchFast()
            rects = ss.process()
            num_rects = len(rects)
            for idx, (x, y, w, h) in enumerate(rects[:1000]):
                score = 1.0 - (float(idx) / float(max(1, num_rects)))
                raw_candidates.append((x, y, x + w, y + h, score))
        except Exception:
            pass

    # 2. Documented Fallback: Multi-cue image processing
    # 2a. MSER (Maximally Stable Extremal Regions)
    try:
        mser = cv2.MSER_create(
            min_area=max(30, int(img_area * min_area_ratio)),
            max_area=int(img_area * max_area_ratio),
        )
        _, bboxes = mser.detectRegions(gray)
        for x, y, w, h in bboxes:
            raw_candidates.append((x, y, x + w, y + h, 0.85))
    except Exception:
        pass

    # 2b. Multi-scale Canny Edges + Morphology
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    for low_t, high_t in [(30, 90), (50, 150), (80, 200)]:
        edges = cv2.Canny(blurred, low_t, high_t)
        for kw, kh in [(15, 7), (7, 15), (11, 11), (21, 9)]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                raw_candidates.append((x, y, x + w, y + h, 0.75))

    # 2c. Adaptive Thresholding Contours
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        raw_candidates.append((x, y, x + w, y + h, 0.70))

    # 2d. Multi-scale spatial grid anchors
    for scale in [0.25, 0.35, 0.50, 0.65, 0.80]:
        for aspect in [0.5, 0.8, 1.2, 1.6, 2.2]:
            bw = int(orig_w * scale)
            bh = int(bw / aspect)
            if bh > orig_h:
                bh = int(orig_h * scale)
                bw = int(bh * aspect)
            step_x = max(16, int(bw * 0.40))
            step_y = max(16, int(bh * 0.40))
            for y in range(0, max(1, orig_h - bh + 1), step_y):
                for x in range(0, max(1, orig_w - bw + 1), step_x):
                    raw_candidates.append((x, y, x + bw, y + bh, 0.50))

    # 3. Coordinate clipping and boundary filtering
    valid_candidates: List[Tuple[Tuple[int, int, int, int], float]] = []
    for x1, y1, x2, y2, score in raw_candidates:
        cx1 = max(0, min(orig_w, int(round(x1))))
        cy1 = max(0, min(orig_h, int(round(y1))))
        cx2 = max(0, min(orig_w, int(round(x2))))
        cy2 = max(0, min(orig_h, int(round(y2))))
        bw = cx2 - cx1
        bh = cy2 - cy1
        if bw < min_size or bh < min_size:
            continue
        area = bw * bh
        ratio = area / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        valid_candidates.append(((cx1, cy1, cx2, cy2), float(score)))

    # 4. Sort descending by candidate score
    valid_candidates.sort(key=lambda r: r[1], reverse=True)

    # 5. IoU deduplication / non-maximum suppression
    kept: List[Tuple[Tuple[int, int, int, int], float]] = []
    for box, score in valid_candidates:
        if all(box_iou(box, prior_box) < iou_thresh for prior_box, _ in kept):
            kept.append((box, score))
            if len(kept) >= limit:
                break
    return kept


class ReferenceMatcher:
    """Training-free region proposals plus frozen DINOv2 exemplar matching.

    SAM (facebook/sam-vit-base) proposes class-neutral candidate segments.
    Their bounding boxes are compared with approved annotation crops using
    frozen DINOv2 (facebook/dinov2-small) cosine similarity features.

    No object detector, semantic head, or fine-tuned model is used in this
    visual-reference path. SAM produces no class labels; DINOv2 produces no
    class predictions — matching is purely non-parametric exemplar comparison.

    Design properties:
    - Similarity scores are cosine similarities in [−1, 1], not calibrated
      probabilities. They are clearly labelled as 'score' (similarity) in
      all returned dictionaries.
    - Per-class scoring averages the K strongest reference similarities (top_k=3)
      so that classes with more approved examples do not automatically win.
    - A configurable margin between best and second-best class guards against
      ambiguous proposals (default margin=0.05; set to 0.0 to disable).
    - Embedding cache is keyed by SHA-1 content hash. Call
      invalidate_reference_cache() when the reference bank changes.
    - SAM and DINOv2 are loaded lazily and offloaded to CPU after each use so
      they never co-occupy GPU memory (4 GB GPU budget).
    - No ML model objects or tensors are ever stored in workspace data files.
    """

    SAM_MODEL_ID = "facebook/sam-vit-base"
    DINO_MODEL_ID = "facebook/dinov2-small"
    # Maximum number of cached embeddings (each ~384 floats ≈ 1.5 KB)
    CACHE_LIMIT = 4096
    # Top-K references per class for averaging (prevents count advantage)
    TOP_K_REFS = 3
    # Minimum pixel size for a valid proposal (both dimensions)
    MIN_PROPOSAL_PX = 16
    # SAM mask quality lower bound; masks below this are dropped before NMS
    MIN_SAM_QUALITY = 0.70
    # Proposal area bounds as fraction of image area
    MIN_AREA_RATIO = 0.0003
    MAX_AREA_RATIO = 0.95
    # NMS IoU thresholds
    SAM_NMS_IOU = 0.82   # de-duplication of SAM proposals
    DET_NMS_IOU = 0.55   # de-duplication of final detections per class
    # Default margin between best and second-best class similarity score
    DEFAULT_MARGIN = 0.05

    def __init__(self, device: str = "auto"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Visual reference matching requires PyTorch and Transformers. "
                "Install model dependencies with: pip install -r requirements-model.txt"
            ) from exc

        from config import resolve_device

        self.device_str, self.device_status = resolve_device(device, torch)
        if self.device_str is None:
            raise RuntimeError(self.device_status)
        self.torch = torch
        self.device = torch.device(self.device_str)
        self.sam_pipeline = None
        self.feature_model = None
        # Embedding cache keyed by SHA-1 content hash.
        # Values are CPU float tensors — never serialized to disk.
        self.embedding_cache: Dict[str, Any] = {}

    def invalidate_reference_cache(self) -> None:
        """Clear cached reference embeddings.

        Call this whenever the reference bank changes (box approved, rejected,
        or manually added) so that the next predict() call re-encodes all
        references. Query-crop embeddings are also cleared for simplicity.
        """
        self.embedding_cache.clear()

    # ------------------------------------------------------------------
    # SAM: class-neutral region proposal generation
    # ------------------------------------------------------------------

    def _load_sam(self) -> Any:
        """Lazily load the SAM mask-generation pipeline."""
        if self.sam_pipeline is None:
            try:
                from transformers import pipeline
                kwargs: Dict[str, Any] = {"device": 0 if self.device_str == "cuda" else -1}
                self.sam_pipeline = pipeline(
                    task="mask-generation", model=self.SAM_MODEL_ID, **kwargs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load SAM class-neutral region proposals ({self.SAM_MODEL_ID}): {exc}. "
                    "The first run downloads ~375 MB. Try CPU mode if CUDA runs out of memory."
                ) from exc
        return self.sam_pipeline

    def _move_sam_to_device(self, device: Any) -> None:
        """Move the SAM model to a device (for sequential GPU sharing)."""
        if self.sam_pipeline is not None:
            try:
                self.sam_pipeline.model.to(device)
            except Exception:
                pass

    def _parse_sam_output(
        self,
        output: Any,
        image_w: int,
        image_h: int,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Extract filtered (box, quality_score) pairs from SAM pipeline output.

        Handles:
        1. Current HF mask-generation pipeline dict format:
           {'masks': [Tensor(H, W), ...], 'scores': Tensor([score, ...])}
        2. HF list-of-dicts format:
           [{'segmentation': ..., 'bbox': [x, y, w, h], 'predicted_iou': ...}, ...]
        3. Legacy boxes format:
           {'bounding_boxes'/'boxes': ..., 'scores': ...}
        """
        import numpy as np

        candidates: List[Tuple[Tuple[int, int, int, int], float]] = []
        raw_masks: List[Any] = []
        scores_arr: Any = None

        if isinstance(output, dict):
            for key in ("masks", "generated_masks", "output"):
                if key in output and (isinstance(output[key], (list, tuple)) or hasattr(output[key], "shape")):
                    raw_masks = list(output[key])
                    break

            if "scores" in output:
                scores_raw = output["scores"]
                if hasattr(scores_raw, "detach"):
                    scores_arr = scores_raw.detach().cpu().numpy()
                else:
                    scores_arr = np.asarray(scores_raw, dtype=float)

            # Fallback: legacy bounding_boxes / boxes tensor format
            if not raw_masks and any(k in output for k in ("bounding_boxes", "boxes")):
                boxes_raw = output.get("bounding_boxes", output.get("boxes", []))
                if hasattr(boxes_raw, "detach"):
                    boxes_raw = boxes_raw.detach().cpu().numpy()
                boxes_arr = (np.asarray(boxes_raw, dtype=float).reshape(-1, 4)
                             if len(boxes_raw) else np.empty((0, 4)))
                sc_arr = scores_arr if scores_arr is not None else np.ones(len(boxes_arr))
                for idx, raw_box in enumerate(boxes_arr):
                    score = float(sc_arr[idx]) if idx < len(sc_arr) else 1.0
                    if score < self.MIN_SAM_QUALITY:
                        continue
                    x1 = max(0, min(image_w, int(round(raw_box[0]))))
                    y1 = max(0, min(image_h, int(round(raw_box[1]))))
                    x2 = max(0, min(image_w, int(round(raw_box[2]))))
                    y2 = max(0, min(image_h, int(round(raw_box[3]))))
                    area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h)
                    if (x2 - x1 < self.MIN_PROPOSAL_PX or y2 - y1 < self.MIN_PROPOSAL_PX
                            or not (self.MIN_AREA_RATIO <= area_ratio <= self.MAX_AREA_RATIO)):
                        continue
                    candidates.append(((x1, y1, x2, y2), score))
                return candidates

        elif isinstance(output, (list, tuple)):
            raw_masks = list(output)

        for idx, item in enumerate(raw_masks):
            score = 1.0
            if scores_arr is not None and idx < len(scores_arr):
                score = float(scores_arr[idx])

            if isinstance(item, dict):
                score = float(
                    item.get("predicted_iou",
                    item.get("iou_score",
                    item.get("stability_score", score)))
                )
                if score < self.MIN_SAM_QUALITY:
                    continue

                bbox = item.get("bbox", item.get("box", None))
                if bbox is not None and len(bbox) == 4:
                    bx, by, bw, bh = [float(v) for v in bbox]
                    x1 = max(0, min(image_w, int(round(bx))))
                    y1 = max(0, min(image_h, int(round(by))))
                    x2 = max(0, min(image_w, int(round(bx + bw))))
                    y2 = max(0, min(image_h, int(round(by + bh))))
                else:
                    seg = item.get("segmentation", None)
                    if seg is None:
                        continue
                    if hasattr(seg, "detach"):
                        mask_arr = seg.detach().cpu().numpy()
                    else:
                        mask_arr = np.asarray(seg, dtype=bool)
                    rows = np.any(mask_arr, axis=1)
                    cols = np.any(mask_arr, axis=0)
                    if not rows.any() or not cols.any():
                        continue
                    r_idx = np.where(rows)[0]
                    c_idx = np.where(cols)[0]
                    x1 = max(0, min(image_w, int(c_idx[0])))
                    y1 = max(0, min(image_h, int(r_idx[0])))
                    x2 = max(0, min(image_w, int(c_idx[-1]) + 1))
                    y2 = max(0, min(image_h, int(r_idx[-1]) + 1))
            else:
                if score < self.MIN_SAM_QUALITY:
                    continue
                if hasattr(item, "detach"):
                    mask_arr = item.detach().cpu().numpy()
                else:
                    mask_arr = np.asarray(item, dtype=bool)
                rows = np.any(mask_arr, axis=1)
                cols = np.any(mask_arr, axis=0)
                if not rows.any() or not cols.any():
                    continue
                r_idx = np.where(rows)[0]
                c_idx = np.where(cols)[0]
                x1 = max(0, min(image_w, int(c_idx[0])))
                y1 = max(0, min(image_h, int(r_idx[0])))
                x2 = max(0, min(image_w, int(c_idx[-1]) + 1))
                y2 = max(0, min(image_h, int(r_idx[-1]) + 1))

            if x2 - x1 < self.MIN_PROPOSAL_PX or y2 - y1 < self.MIN_PROPOSAL_PX:
                continue
            area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h)
            if not (self.MIN_AREA_RATIO <= area_ratio <= self.MAX_AREA_RATIO):
                continue

            candidates.append(((x1, y1, x2, y2), score))

        return candidates

    def _proposals_sam(
        self, image: Image.Image, limit: int = 240
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Generate class-neutral candidate bounding boxes using SAM automatic masks.

        SAM generates masks without any class labels or semantic meaning.
        Boxes are filtered by mask quality score, minimum size, area ratio,
        then deduplicated with NMS. Returns up to `limit` candidates sorted
        by descending quality score.
        """
        rgb_image = image.convert("RGB")
        orig_w, orig_h = rgb_image.size

        # Scale down oversized images for proposal generation (10x faster on GPU, low VRAM)
        scale = 1.0
        if max(orig_w, orig_h) > 800:
            scale = 800.0 / float(max(orig_w, orig_h))
            proc_w = max(16, int(round(orig_w * scale)))
            proc_h = max(16, int(round(orig_h * scale)))
            sam_image = rgb_image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
        else:
            sam_image = rgb_image
            proc_w, proc_h = orig_w, orig_h

        pipe = self._load_sam()
        try:
            output = pipe(
                sam_image,
                points_per_batch=16 if self.device_str == "cuda" else 8,
                points_per_crop=16,
                pred_iou_thresh=self.MIN_SAM_QUALITY,
                stability_score_thresh=0.88,
                crops_n_layers=0,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if self.device_str == "cuda":
                self.torch.cuda.empty_cache()
            # Retry with minimal points batch
            try:
                output = pipe(
                    sam_image,
                    points_per_batch=1,
                    points_per_crop=16,
                    pred_iou_thresh=self.MIN_SAM_QUALITY,
                    stability_score_thresh=0.88,
                    crops_n_layers=0,
                )
            except Exception as retry_exc:
                raise RuntimeError(
                    "CUDA ran out of memory while proposing candidate regions. "
                    "Select CPU in Compute Device and retry."
                ) from retry_exc

        candidates = self._parse_sam_output(output, proc_w, proc_h)

        # Rescale coordinates back to original image size if resized
        if scale != 1.0:
            rescaled_candidates: List[Tuple[Tuple[int, int, int, int], float]] = []
            for (bx1, by1, bx2, by2), score in candidates:
                rx1 = max(0, min(orig_w, int(round(bx1 / scale))))
                ry1 = max(0, min(orig_h, int(round(by1 / scale))))
                rx2 = max(0, min(orig_w, int(round(bx2 / scale))))
                ry2 = max(0, min(orig_h, int(round(by2 / scale))))
                if rx2 > rx1 and ry2 > ry1:
                    rescaled_candidates.append(((rx1, ry1, rx2, ry2), score))
            candidates = rescaled_candidates

        # Sort by quality score descending, then apply NMS deduplication
        candidates.sort(key=lambda row: row[1], reverse=True)
        kept: List[Tuple[Tuple[int, int, int, int], float]] = []
        for box, score in candidates:
            if all(box_iou(box, prior_box) < self.SAM_NMS_IOU for prior_box, _ in kept):
                kept.append((box, score))
                if len(kept) >= limit:
                    break
        return kept

    def _proposals_opencv(
        self, image: Image.Image, limit: int = 240
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Generate class-neutral candidate bounding boxes using OpenCV image processing."""
        return generate_opencv_proposals(
            image,
            limit=limit,
            min_size=self.MIN_PROPOSAL_PX,
            min_area_ratio=self.MIN_AREA_RATIO,
            max_area_ratio=self.MAX_AREA_RATIO,
            iou_thresh=self.SAM_NMS_IOU,
        )

    def proposals(
        self, image: Image.Image, limit: int = 240, method: str = "opencv"
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Generate class-neutral candidate bounding boxes using OpenCV or SAM.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to generate candidate regions for.
        limit : int
            Maximum number of candidates to return.
        method : str, default 'opencv'
            Proposal generator method: 'opencv' (fast image processing) or 'sam' (deep visual masks).
        """
        if str(method).strip().lower() == "sam":
            return self._proposals_sam(image, limit=limit)
        return self._proposals_opencv(image, limit=limit)

    # ------------------------------------------------------------------
    # Exemplar feature encoding
    # ------------------------------------------------------------------

    def _load_feature_model(self) -> Any:
        """Lazily load the frozen feature encoder onto the compute device."""
        if self.feature_model is None:
            try:
                from transformers import AutoModel
                self.feature_model = AutoModel.from_pretrained(self.DINO_MODEL_ID)
                self.feature_model.eval()
                # Strictly freeze: no gradient computation, no weight updates
                for param in self.feature_model.parameters():
                    param.requires_grad = False
                self.feature_model.to(self.device)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load the frozen feature encoder ({self.DINO_MODEL_ID}): {exc}. "
                    "The first run downloads ~85 MB."
                ) from exc
        return self.feature_model

    @staticmethod
    def _prepare_crop(image: Image.Image) -> Any:
        """Resize with padding to 224×224, preserving aspect ratio without cropping."""
        import numpy as np
        from PIL import ImageOps

        padded = ImageOps.pad(image.convert("RGB"), (224, 224), color=(124, 116, 104))
        array = np.asarray(padded, dtype=np.float32) / 255.0
        tensor = __import__("torch").from_numpy(array.transpose(2, 0, 1).copy())
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (tensor - mean) / std

    @staticmethod
    def _crop_key(image: Image.Image) -> str:
        """SHA-1 content hash for cache lookup."""
        import hashlib
        rgb = image.convert("RGB")
        return hashlib.sha1(str(rgb.size).encode("ascii") + rgb.tobytes()).hexdigest()

    def encode(self, images: Sequence[Image.Image]) -> Any:
        """Encode crops with frozen visual features on GPU, using an in-memory cache.

        Embeddings are stored in-memory and are never serialized to disk.
        """
        torch = self.torch
        model = self._load_feature_model()
        keys = [self._crop_key(im) for im in images]
        missing_indices = [i for i, key in enumerate(keys) if key not in self.embedding_cache]
        with torch.inference_mode():
            batch_size = 16 if self.device_str == "cuda" else 8
            for start in range(0, len(missing_indices), batch_size):
                indices = missing_indices[start: start + batch_size]
                batch = torch.stack([self._prepare_crop(images[i]) for i in indices]).to(self.device)
                outputs = model(pixel_values=batch)
                # CLS token (position 0) — global image representation
                features = torch.nn.functional.normalize(
                    outputs.last_hidden_state[:, 0].float(), dim=-1
                )
                for local_idx, image_idx in enumerate(indices):
                    self.embedding_cache[keys[image_idx]] = features[local_idx].detach().cpu()
            # Evict oldest entries when cache exceeds limit
            while len(self.embedding_cache) > self.CACHE_LIMIT:
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            return torch.stack([self.embedding_cache[key] for key in keys]).to(self.device)

    # ------------------------------------------------------------------
    # Scoring: top-K averaging with margin guard
    # ------------------------------------------------------------------

    def _score_proposals(
        self,
        similarities: Any,
        ref_classes: List[str],
        class_names: List[str],
        top_k: int,
    ) -> Any:
        """Compute per-class similarity scores for each proposal.

        Each class is scored by averaging its strongest `top_k` reference
        similarities per proposal.  This prevents classes with more approved
        examples from systematically outscoring smaller classes.

        Returns np.ndarray of shape (N_proposals, N_classes) — cosine
        similarity scores, NOT calibrated probabilities.
        """
        import numpy as np

        class_scores = []
        for class_name in class_names:
            indices = [i for i, name in enumerate(ref_classes) if name == class_name]
            sims = similarities[:, indices]           # (N_proposals, N_class_refs)
            k = min(top_k, sims.shape[1])
            top_sims = np.sort(sims, axis=1)[:, -k:]  # k largest per proposal
            class_scores.append(top_sims.mean(axis=1))

        return np.stack(class_scores, axis=1)         # (N_proposals, N_classes)

    # ------------------------------------------------------------------
    # Main predict entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Image.Image,
        references: Dict[str, List[Image.Image]],
        threshold: float = 0.45,
        limit: int = 240,
        margin: float = DEFAULT_MARGIN,
        car_nms_iou: float = 0.5,
        proposal_method: str = "opencv",
    ) -> List[Dict[str, Any]]:
        """Match candidate regions to reference crops via frozen DINOv2 cosine similarity.

        Workflow:
        1. Propose class-neutral candidate boxes using either OpenCV or SAM.
        2. Each proposal crop and reference crop is encoded with frozen DINOv2.
        3. Cosine similarity matrix (proposals × references) is computed.
        4. Per-class scores = average of top-K reference similarities per class.
        5. A proposal is assigned to a class only when:
           a) Its class score >= `threshold` (similarity cutoff, not a probability).
           b) Its score exceeds the second-best class by >= `margin`.
        6. Per-class NMS removes spatially redundant detections.

        Returned 'score' values are cosine similarities, not calibrated probabilities.

        Parameters
        ----------
        image : PIL.Image.Image
            Query image to scan.
        references : dict mapping class_name -> list of approved PIL crop images.
        threshold : float
            Minimum cosine similarity for a proposal to be returned.
        limit : int
            Maximum number of candidate proposals to evaluate.
        margin : float
            Minimum similarity advantage over the second-best class.
            Set to 0.0 to disable the margin guard.
        car_nms_iou : float
            IoU threshold for car deduplication.
        proposal_method : str, default 'opencv'
            Proposal generator to use ('opencv' or 'sam').

        Returns
        -------
        list of dicts with keys:
            'class'          : assigned class name
            'box'            : [x1, y1, x2, y2] pixel coordinates
            'score'          : cosine similarity (float, NOT a probability)
            'proposal_score' : candidate region quality score
            'source'         : 'auto'
        """
        import numpy as np

        active_references = {name: crops for name, crops in references.items() if crops}
        if not active_references:
            return []

        proposals = self.proposals(image, limit=limit, method=proposal_method)
        if not proposals:
            return []

        boxes = [box for box, _ in proposals]
        proposal_quality = [score for _, score in proposals]

        ref_images: List[Image.Image] = []
        ref_classes: List[str] = []
        for class_name, crops in active_references.items():
            for crop in crops:
                ref_images.append(crop)
                ref_classes.append(class_name)

        class_names = list(active_references.keys())

        # Encode with frozen DINOv2 (SAM already offloaded to CPU above)
        refs_emb = self.encode(ref_images)           # (N_refs, D)
        query_images = [crop_box(image, box) for box in boxes]
        queries_emb = self.encode(query_images)       # (N_proposals, D)

        # Cosine similarity matrix (both sides are L2-normalized in encode())
        similarities = (queries_emb @ refs_emb.T).detach().cpu().numpy()

        # Per-class score matrix: (N_proposals, N_classes)
        score_matrix = self._score_proposals(
            similarities, ref_classes, class_names, top_k=self.TOP_K_REFS
        )

        best_class_indices = score_matrix.argmax(axis=1)
        best_scores = score_matrix.max(axis=1)

        has_multi_class = len(class_names) > 1
        if has_multi_class:
            second_matrix = score_matrix.copy()
            second_matrix[
                np.arange(len(best_class_indices)), best_class_indices
            ] = -np.inf
            second_best_scores = second_matrix.max(axis=1)
        else:
            second_best_scores = np.full_like(best_scores, -np.inf)

        detections: List[Dict[str, Any]] = []
        for class_idx, class_name in enumerate(class_names):
            indices = [
                i for i, score in enumerate(best_scores)
                if (score >= threshold
                    and best_class_indices[i] == class_idx
                    and (not has_multi_class or (score - second_best_scores[i]) >= margin))
            ]
            indices.sort(key=lambda i: best_scores[i], reverse=True)

            accepted: List[int] = []
            for i in indices:
                if all(box_iou(boxes[i], boxes[j]) < self.DET_NMS_IOU for j in accepted):
                    accepted.append(i)
                    if len(accepted) >= 30:
                        break

            for i in accepted:
                detections.append({
                    "class": class_name,
                    "box": list(boxes[i]),
                    "score": round(float(best_scores[i]), 4),
                    "proposal_score": round(float(proposal_quality[i]), 4),
                    "source": "auto",
                })
        detections = sorted(detections, key=lambda item: item["score"], reverse=True)
        return filter_one_car_per_image(detections)


def filter_one_car_per_image(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enforce strictly one most accurate car box per image.

    Among all predictions for an image, selects the single car box with the highest
    similarity score (most accurate). Ties in score are broken by larger box area.
    Preserves non-car predictions if any exist.
    """
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
        # Primary key: similarity score (descending); Tie-breaker: box area (descending)
        return (score, area)

    best_car = max(car_preds, key=_car_sort_key)
    result = non_car_preds + [best_car]
    return sorted(result, key=lambda p: float(p.get("score", 0.0)), reverse=True)


def deduplicate_car_predictions(
    predictions: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Retain strictly the single most accurate car box per image."""
    return filter_one_car_per_image(predictions)


