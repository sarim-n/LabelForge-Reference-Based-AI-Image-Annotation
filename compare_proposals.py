"""Comparison benchmark between OpenCV and SAM candidate region proposals."""

import json
import time
import os
from PIL import Image
from vision import generate_opencv_proposals, ReferenceMatcher, box_iou

def run_comparison():
    data_path = "work/workspace/project.json"
    with open(data_path, "r", encoding="utf-8") as f:
        proj = json.load(f)

    labels = proj.get("labels", {})
    # Select 5 representative images that have ground truth labels
    sample_images = ["car_001.jpg", "car_002.jpg", "car_003.jpg", "car_004.jpg", "car_005.jpg"]
    # Check if they exist on disk
    valid_samples = [
        name for name in sample_images
        if os.path.exists(os.path.join("work/workspace/images", name)) and labels.get(name)
    ]
    if not valid_samples:
        valid_samples = [
            name for name in labels.keys()
            if os.path.exists(os.path.join("work/workspace/images", name)) and labels.get(name)
        ][:5]

    print(f"Running comparison on {len(valid_samples)} representative labeled images:")
    for name in valid_samples:
        gt_boxes = [item["box"] for item in labels[name]]
        print(f" - {name}: {len(gt_boxes)} ground-truth labeled box(es)")

    matcher = ReferenceMatcher(device="CUDA")

    opencv_times = []
    opencv_box_counts = []
    opencv_recalls_50 = []
    opencv_recalls_40 = []

    sam_times = []
    sam_box_counts = []
    sam_recalls_50 = []
    sam_recalls_40 = []

    iou_thresh_50 = 0.50
    iou_thresh_40 = 0.40

    for name in valid_samples:
        img_path = os.path.join("work/workspace/images", name)
        img = Image.open(img_path).convert("RGB")
        gt_boxes = [item["box"] for item in labels[name]]

        # 1. OpenCV proposals
        t0 = time.perf_counter()
        cv_proposals = generate_opencv_proposals(img, limit=240)
        t_cv = time.perf_counter() - t0
        opencv_times.append(t_cv)
        opencv_box_counts.append(len(cv_proposals))

        cv_boxes = [b for b, _ in cv_proposals]
        hit_50 = sum(
            1 for gt in gt_boxes if any(box_iou(gt, prop) >= iou_thresh_50 for prop in cv_boxes)
        )
        hit_40 = sum(
            1 for gt in gt_boxes if any(box_iou(gt, prop) >= iou_thresh_40 for prop in cv_boxes)
        )
        opencv_recalls_50.append(hit_50 / max(1, len(gt_boxes)))
        opencv_recalls_40.append(hit_40 / max(1, len(gt_boxes)))

        # 2. SAM proposals
        t0 = time.perf_counter()
        sam_proposals = matcher._proposals_sam(img, limit=240) if hasattr(matcher, "_proposals_sam") else matcher.proposals(img, limit=240)
        t_sam = time.perf_counter() - t0
        sam_times.append(t_sam)
        sam_box_counts.append(len(sam_proposals))

        sam_boxes = [b for b, _ in sam_proposals]
        hit_sam_50 = sum(
            1 for gt in gt_boxes if any(box_iou(gt, prop) >= iou_thresh_50 for prop in sam_boxes)
        )
        hit_sam_40 = sum(
            1 for gt in gt_boxes if any(box_iou(gt, prop) >= iou_thresh_40 for prop in sam_boxes)
        )
        sam_recalls_50.append(hit_sam_50 / max(1, len(gt_boxes)))
        sam_recalls_40.append(hit_sam_40 / max(1, len(gt_boxes)))

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS: OpenCV vs SAM Candidate Region Proposals")
    print("=" * 80)
    print(f"{'Image':<14} | {'OpenCV Boxes':<12} | {'SAM Boxes':<10} | {'OpenCV Time':<12} | {'SAM Time':<10} | {'OpenCV Hit (0.50)':<16} | {'SAM Hit (0.50)'}")
    print("-" * 80)
    for idx, name in enumerate(valid_samples):
        print(f"{name:<14} | {opencv_box_counts[idx]:<12} | {sam_box_counts[idx]:<10} | {opencv_times[idx]:.3f}s{'':<6} | {sam_times[idx]:.3f}s{'':<4} | {opencv_recalls_50[idx]*100:.1f}%{'':<11} | {sam_recalls_50[idx]*100:.1f}%")

    print("-" * 80)
    avg_cv_boxes = sum(opencv_box_counts) / len(opencv_box_counts)
    avg_sam_boxes = sum(sam_box_counts) / len(sam_box_counts)
    avg_cv_time = sum(opencv_times) / len(opencv_times)
    avg_sam_time = sum(sam_times) / len(sam_times)
    avg_cv_recall_50 = sum(opencv_recalls_50) / len(opencv_recalls_50)
    avg_sam_recall_50 = sum(sam_recalls_50) / len(sam_recalls_50)
    avg_cv_recall_40 = sum(opencv_recalls_40) / len(opencv_recalls_40)
    avg_sam_recall_40 = sum(sam_recalls_40) / len(sam_recalls_40)

    print(f"{'AVERAGE':<14} | {avg_cv_boxes:<12.1f} | {avg_sam_boxes:<10.1f} | {avg_cv_time:.3f}s{'':<6} | {avg_sam_time:.3f}s{'':<4} | {avg_cv_recall_50*100:.1f}%{'':<11} | {avg_sam_recall_50*100:.1f}%")
    print(f"Average Hit Rate (IoU >= 0.40): OpenCV: {avg_cv_recall_40*100:.1f}% | SAM: {avg_sam_recall_40*100:.1f}%")
    print(f"Speedup Factor: OpenCV is {avg_sam_time / max(0.001, avg_cv_time):.1f}x faster per image than SAM.")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
