"""Focused test suite for OpenCV candidate region generation and DINOv2 reference matcher integration."""

import unittest
import numpy as np
from PIL import Image

from vision import (
    generate_opencv_proposals,
    ReferenceMatcher,
    box_iou,
    crop_box,
    yolo_line,
    split_dataset,
    validate_dataset,
)
from config import PROPOSAL_OPENCV, PROPOSAL_SAM, PROPOSAL_METHOD_CHOICES


class TestOpenCVProposals(unittest.TestCase):
    """Test suite verifying OpenCV candidate generation, filtering, clipping, and matching."""

    def setUp(self):
        # Create a test image with distinct geometric patterns and high contrast objects
        self.img_w, self.img_h = 400, 300
        arr = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 220
        # Draw a simulated vehicle / rectangle object
        arr[80:200, 100:300, :] = 40
        # Draw a small contrasting badge/detail inside
        arr[120:160, 150:250, :] = 200
        self.test_img = Image.fromarray(arr)

    def test_generate_opencv_proposals_structure_and_limit(self):
        """Verify output format, tuple structure, and limit cap."""
        limit = 50
        proposals = generate_opencv_proposals(self.test_img, limit=limit)
        self.assertIsInstance(proposals, list)
        self.assertGreater(len(proposals), 0)
        self.assertLessEqual(len(proposals), limit)

        for item in proposals:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            box, score = item
            self.assertIsInstance(box, tuple)
            self.assertEqual(len(box), 4)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_coordinate_clipping(self):
        """Verify all proposed boxes are strictly clipped within image bounds."""
        proposals = generate_opencv_proposals(self.test_img, limit=100)
        for (x1, y1, x2, y2), _ in proposals:
            self.assertGreaterEqual(x1, 0, f"x1 ({x1}) < 0")
            self.assertGreaterEqual(y1, 0, f"y1 ({y1}) < 0")
            self.assertLessEqual(x2, self.img_w, f"x2 ({x2}) > img_w ({self.img_w})")
            self.assertLessEqual(y2, self.img_h, f"y2 ({y2}) > img_h ({self.img_h})")
            self.assertLess(x1, x2, f"Invalid box: x1 ({x1}) >= x2 ({x2})")
            self.assertLess(y1, y2, f"Invalid box: y1 ({y1}) >= y2 ({y2})")

    def test_invalid_and_tiny_box_filtering(self):
        """Verify boxes below min_size or outside area ratio limits are dropped."""
        min_size = 20
        min_area_ratio = 0.01
        max_area_ratio = 0.90
        img_area = float(self.img_w * self.img_h)

        proposals = generate_opencv_proposals(
            self.test_img,
            limit=100,
            min_size=min_size,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        for (x1, y1, x2, y2), _ in proposals:
            bw = x2 - x1
            bh = y2 - y1
            self.assertGreaterEqual(bw, min_size)
            self.assertGreaterEqual(bh, min_size)
            area = bw * bh
            ratio = area / img_area
            self.assertGreaterEqual(ratio, min_area_ratio)
            self.assertLessEqual(ratio, max_area_ratio)

    def test_duplicate_suppression(self):
        """Verify no two proposals exceed the IoU deduplication threshold."""
        iou_thresh = 0.80
        proposals = generate_opencv_proposals(self.test_img, limit=100, iou_thresh=iou_thresh)
        boxes = [b for b, _ in proposals]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = box_iou(boxes[i], boxes[j])
                self.assertLess(
                    iou,
                    iou_thresh + 1e-5,
                    f"Duplicate boxes not suppressed: {boxes[i]} vs {boxes[j]} with IoU {iou:.3f}",
                )

    def test_reference_matcher_opencv_dispatch(self):
        """Verify ReferenceMatcher.proposals correctly dispatches to OpenCV generator."""
        matcher = ReferenceMatcher(device="auto")
        proposals_cv = matcher.proposals(self.test_img, limit=30, method="opencv")
        self.assertIsInstance(proposals_cv, list)
        self.assertGreater(len(proposals_cv), 0)
        self.assertLessEqual(len(proposals_cv), 30)

    def test_dinov2_matching_end_to_end_with_opencv(self):
        """Verify ReferenceMatcher.predict runs end-to-end with OpenCV proposals and frozen DINOv2."""
        matcher = ReferenceMatcher(device="auto")
        # Crop an exemplar reference
        ref_crop = crop_box(self.test_img, [100, 80, 300, 200])
        refs = {"vehicle": [ref_crop]}

        # Predict with proposal_method='opencv'
        preds = matcher.predict(
            self.test_img,
            refs,
            threshold=0.20,
            limit=50,
            proposal_method="opencv",
        )
        self.assertIsInstance(preds, list)
        if preds:
            first = preds[0]
            self.assertEqual(first["class"], "vehicle")
            self.assertIn("score", first)
            self.assertIn("proposal_score", first)
            self.assertIn("box", first)
            self.assertEqual(len(first["box"]), 4)
            self.assertEqual(first["source"], "auto")

    def test_config_choices_preserved(self):
        """Verify proposal configuration choices are defined and clear."""
        self.assertEqual(PROPOSAL_OPENCV, "OpenCV (Fast / Lightweight)")
        self.assertEqual(PROPOSAL_SAM, "Deep Visual Masks (High Detail)")
        self.assertIn(PROPOSAL_OPENCV, PROPOSAL_METHOD_CHOICES)
        self.assertIn(PROPOSAL_SAM, PROPOSAL_METHOD_CHOICES)

    def test_existing_utilities_preserved(self):
        """Verify existing dataset and annotation utilities function unchanged."""
        line = yolo_line(0, [50, 50, 150, 150], 200, 200)
        self.assertTrue(line.startswith("0 "))
        parts = [float(p) for p in line.strip().split()]
        self.assertEqual(len(parts), 5)
        self.assertAlmostEqual(parts[1], 0.5)
        self.assertAlmostEqual(parts[2], 0.5)

        # Dataset validation
        val_res = validate_dataset(
            {"img1.jpg": self.test_img},
            {"img1.jpg": [{"box": [10, 10, 50, 50], "class": "car"}]},
            ["car"],
        )
        self.assertTrue(val_res["is_valid"])


if __name__ == "__main__":
    unittest.main()
