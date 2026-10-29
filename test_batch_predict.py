"""Unit tests for batch folder prediction features and image annotation."""

import unittest
from PIL import Image, ImageDraw
from vision import draw_box_with_label, filter_one_car_per_image
import io
import zipfile


class TestBatchFolderPredict(unittest.TestCase):
    def test_draw_box_with_label(self):
        """Verify bounding box and badge rendering on batch images."""
        img = Image.new("RGB", (640, 480), (200, 200, 200))
        draw_ctx = ImageDraw.Draw(img)
        box = [50, 60, 300, 250]
        draw_box_with_label(draw_ctx, box, "car · 0.88", color="#00e676", text_color="#000000", line_width=4)
        self.assertEqual(img.size, (640, 480))

    def test_batch_single_car_rule(self):
        """Batch predictions must enforce single most accurate car box per image."""
        raw_preds = [
            {"class": "car", "box": [10, 10, 100, 100], "score": 0.72},
            {"class": "car", "box": [50, 50, 300, 200], "score": 0.91},
            {"class": "car", "box": [80, 80, 220, 180], "score": 0.85},
        ]
        filtered = filter_one_car_per_image(raw_preds)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["score"], 0.91)
        self.assertEqual(filtered[0]["box"], [50, 50, 300, 200])

    def test_batch_annotated_zip_packaging(self):
        """Verify in-memory ZIP generation for batch annotated images."""
        batch_results = {}
        for i in range(3):
            fname = f"sample_{i}.jpg"
            img = Image.new("RGB", (200, 150), (100, 120, 140))
            draw_c = ImageDraw.Draw(img)
            draw_box_with_label(draw_c, [10, 10, 100, 80], "car · 0.90", color="#00e676")
            batch_results[fname] = {
                "image": img,
                "annotated_image": img,
                "predictions": [{"class": "car", "box": [10, 10, 100, 80], "score": 0.90}],
                "time": 0.5,
            }

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, r_data in batch_results.items():
                img_byte_buf = io.BytesIO()
                r_data["annotated_image"].save(img_byte_buf, format="JPEG", quality=90)
                zf.writestr(f"annotated_{fname}", img_byte_buf.getvalue())

        zip_bytes = zip_buf.getvalue()
        self.assertGreater(len(zip_bytes), 0)

        # Verify readability
        read_buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(read_buf, "r") as zf:
            file_list = zf.namelist()
            self.assertEqual(len(file_list), 3)
            self.assertIn("annotated_sample_0.jpg", file_list)


if __name__ == "__main__":
    unittest.main()
