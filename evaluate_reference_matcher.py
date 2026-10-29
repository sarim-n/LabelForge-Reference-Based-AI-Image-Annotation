"""Evaluation benchmark for the Reference-Based Auto-Labeling workflow.

Measures precision, recall, F1, box IoU quality, false positives, missed objects,
and processing throughput on held-out images.

Guarantees strict separation between reference exemplars and evaluation images
to prevent train/reference leakage.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

from vision import ReferenceMatcher, box_iou, crop_box


def load_labeled_dataset(
    workspace_dir: str = "work/workspace",
) -> Tuple[Dict[str, Image.Image], Dict[str, List[Dict[str, Any]]]]:
    """Load images and ground-truth annotations from the workspace."""
    ws_path = Path(workspace_dir)
    project_file = ws_path / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"Project manifest not found at {project_file}")

    with open(project_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    images: Dict[str, Image.Image] = {}
    for img_meta in manifest.get("images", []):
        img_name = img_meta["name"]
        rel_path = img_meta.get("path", f"images/{img_name}")
        full_path = ws_path / rel_path
        if full_path.exists():
            images[img_name] = Image.open(full_path).convert("RGB")

    labels: Dict[str, List[Dict[str, Any]]] = manifest.get("labels", {})
    # Filter to only images that exist and have annotations
    valid_labels = {k: v for k, v in labels.items() if k in images and len(v) > 0}
    return images, valid_labels


def split_reference_and_heldout(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    num_reference_images: int = 8,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Partition labeled image names into disjoint reference and held-out test sets.

    Ensures zero leakage: reference crops are derived solely from reference images,
    and evaluation is conducted solely on held-out images.
    """
    sorted_names = sorted(labels.keys())
    rng = np.random.RandomState(seed)
    shuffled = sorted_names.copy()
    rng.shuffle(shuffled)

    num_ref = min(num_reference_images, max(1, len(shuffled) // 4))
    ref_image_names = shuffled[:num_ref]
    test_image_names = shuffled[num_ref:]
    return ref_image_names, test_image_names


def build_leak_free_references(
    images: Dict[str, Image.Image],
    labels: Dict[str, List[Dict[str, Any]]],
    ref_image_names: Sequence[str],
) -> Dict[str, List[Image.Image]]:
    """Build reference crop dictionary using only approved boxes from reference images."""
    references: Dict[str, List[Image.Image]] = {}
    for name in ref_image_names:
        img = images[name]
        for box_entry in labels.get(name, []):
            cls_name = box_entry["class"]
            box = box_entry["box"]
            crop = crop_box(img, box)
            if cls_name not in references:
                references[cls_name] = []
            references[cls_name].append(crop)
    return references


def evaluate_predictions(
    ground_truth_boxes: List[Dict[str, Any]],
    predicted_boxes: List[Dict[str, Any]],
    iou_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Greedy bipartite matching between predictions and ground-truth boxes.

    Returns:
        dict with TP, FP, FN, matched_ious, precision, recall, f1
    """
    # Group by class
    gt_by_class: Dict[str, List[List[float]]] = {}
    for gt in ground_truth_boxes:
        gt_by_class.setdefault(gt["class"], []).append(list(gt["box"]))

    pred_by_class: Dict[str, List[Dict[str, Any]]] = {}
    for p in predicted_boxes:
        pred_by_class.setdefault(p["class"], []).append(p)

    all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: List[float] = []

    for cls in all_classes:
        gts = gt_by_class.get(cls, [])
        preds = pred_by_class.get(cls, [])

        # Sort predictions by score descending
        preds_sorted = sorted(preds, key=lambda x: x.get("score", 0.0), reverse=True)
        matched_gt = [False] * len(gts)

        for p in preds_sorted:
            p_box = p["box"]
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, g_box in enumerate(gts):
                if matched_gt[gt_idx]:
                    continue
                iou = box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                total_tp += 1
                matched_gt[best_gt_idx] = True
                matched_ious.append(best_iou)
            else:
                total_fp += 1

        # Any unmatched GT are false negatives
        total_fn += (len(gts) - sum(matched_gt))

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "matched_ious": matched_ious,
    }


def run_benchmark(
    workspace_dir: str = "work/workspace",
    num_reference_images: int = 8,
    max_test_images: int = 15,
    threshold: float = 0.55,
    margin: float = 0.05,
    iou_thresh: float = 0.50,
    device: str = "auto",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete held-out evaluation for ReferenceMatcher."""
    images, labels = load_labeled_dataset(workspace_dir)
    ref_names, test_names = split_reference_and_heldout(
        images, labels, num_reference_images=num_reference_images, seed=seed
    )

    if max_test_images > 0:
        test_names = test_names[:max_test_images]

    references = build_leak_free_references(images, labels, ref_names)
    total_ref_crops = sum(len(crops) for crops in references.values())

    matcher = ReferenceMatcher(device=device)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: List[float] = []
    per_image_times: List[float] = []
    total_gt_boxes = 0
    total_pred_boxes = 0

    print("=" * 72)
    print("LabelForge Reference-Based Auto-Labeling Benchmark")
    print(f"Device: {matcher.device_str} ({matcher.device_status})")
    print(f"Reference images: {len(ref_names)} ({total_ref_crops} reference crops)")
    print(f"Held-out test images: {len(test_names)} (zero leakage)")
    print(f"Similarity threshold: {threshold:.2f} | Margin: {margin:.2f} | Eval IoU: {iou_thresh:.2f}")
    print("=" * 72)

    for idx, test_name in enumerate(test_names, 1):
        test_img = images[test_name]
        gt_boxes = labels.get(test_name, [])
        total_gt_boxes += len(gt_boxes)

        t0 = time.perf_counter()
        preds = matcher.predict(
            test_img,
            references=references,
            threshold=threshold,
            margin=margin,
        )
        elapsed = time.perf_counter() - t0
        per_image_times.append(elapsed)
        total_pred_boxes += len(preds)

        res = evaluate_predictions(gt_boxes, preds, iou_threshold=iou_thresh)
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]
        all_matched_ious.extend(res["matched_ious"])

        print(
            f"[{idx:02d}/{len(test_names):02d}] {test_name}: "
            f"GT={len(gt_boxes)} Pred={len(preds)} "
            f"(TP={res['tp']} FP={res['fp']} FN={res['fn']}) in {elapsed:.2f}s"
        )

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = (2 * precision * recall) / max(1e-6, precision + recall)
    mean_iou = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

    total_time = sum(per_image_times)
    avg_time = total_time / max(1, len(per_image_times))
    throughput_fps = len(test_names) / max(1e-4, total_time)

    metrics = {
        "device": matcher.device_str,
        "ref_images": len(ref_names),
        "ref_crops": total_ref_crops,
        "test_images": len(test_names),
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "missed_objects_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_box_iou": round(mean_iou, 4),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_image_seconds": round(avg_time, 2),
        "throughput_images_per_sec": round(throughput_fps, 3),
        "threshold": threshold,
        "margin": margin,
        "iou_threshold": iou_thresh,
    }

    print("-" * 72)
    print("BENCHMARK SUMMARY RESULTS:")
    print(f"  Precision:       {metrics['precision']:.4f} ({total_tp}/{total_tp + total_fp})")
    print(f"  Recall:          {metrics['recall']:.4f} ({total_tp}/{total_gt_boxes})")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  Mean Box IoU:    {metrics['mean_box_iou']:.4f} (on true positive matches)")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  Missed Objects:  {metrics['missed_objects_fn']}")
    print(f"  Avg Latency:     {metrics['avg_time_per_image_seconds']:.2f}s per image")
    print(f"  Throughput:      {metrics['throughput_images_per_sec']:.2f} images/sec")
    print("=" * 72)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ReferenceMatcher on held-out images.")
    parser.add_argument("--workspace", default="work/workspace", help="Path to workspace dir")
    parser.add_argument("--num-refs", type=int, default=8, help="Number of reference images")
    parser.add_argument("--max-test", type=int, default=15, help="Max test images (0 for all)")
    parser.add_argument("--thresh", type=float, default=0.55, help="Cosine similarity threshold")
    parser.add_argument("--margin", type=float, default=0.05, help="Ambiguity margin")
    parser.add_argument("--iou", type=float, default=0.50, help="Evaluation IoU threshold")
    parser.add_argument("--device", default="auto", help="Compute device (auto/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed")
    parser.add_argument("--out", default="", help="Optional JSON path to save metrics")
    args = parser.parse_args()

    results = run_benchmark(
        workspace_dir=args.workspace,
        num_reference_images=args.num_refs,
        max_test_images=args.max_test,
        threshold=args.thresh,
        margin=args.margin,
        iou_thresh=args.iou,
        device=args.device,
        seed=args.seed,
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {args.out}")
