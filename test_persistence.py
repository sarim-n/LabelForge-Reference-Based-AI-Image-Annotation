"""Unit tests for LabelForge local workspace persistence using standard library unittest."""

import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from PIL import Image

import persistence
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


def create_sample_image(width=100, height=80, color=(255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_workspace(self):
        ws_dir = self.tmp_path / "test_ws"
        self.assertFalse(has_saved_workspace(ws_dir))

        img1 = create_sample_image(200, 150, (255, 100, 50))
        img2 = create_sample_image(300, 200, (50, 150, 200))
        crop1 = create_sample_image(40, 40, (10, 20, 30))

        images = {"img1.jpg": img1, "img2.png": img2}
        labels = {
            "img1.jpg": [{"class": "cup", "box": [10, 10, 50, 60], "source": "manual"}],
            "img2.png": [{"class": "bottle", "box": [20, 30, 80, 120], "source": "auto", "score": 0.92}],
        }
        refs = {"cup": [crop1]}
        pending = {
            "img1.jpg": [{"class": "bottle", "box": [30, 40, 90, 110], "score": 0.85, "source": "auto", "id": "p1"}]
        }
        times = [{"seconds": 0.45, "images": 2}]
        rejections = [1, 1, 1]
        settings = {"model_choice": "Grounding DINO", "train_pct": 75}

        save_info = save_workspace(
            ws_dir,
            images=images,
            labels=labels,
            refs=refs,
            pending=pending,
            times=times,
            rejections=rejections,
            settings=settings,
        )

        self.assertEqual(save_info["status"], "Saved")
        self.assertTrue(has_saved_workspace(ws_dir))
        self.assertTrue((ws_dir / "project.json").exists())

        # Load and verify
        loaded = load_workspace(ws_dir)
        self.assertEqual(set(loaded["images"].keys()), {"img1.jpg", "img2.png"})
        self.assertEqual(loaded["images"]["img1.jpg"].size, (200, 150))
        self.assertEqual(loaded["images"]["img2.png"].size, (300, 200))

        self.assertEqual(len(loaded["labels"]["img1.jpg"]), 1)
        self.assertEqual(loaded["labels"]["img1.jpg"][0]["class"], "cup")
        self.assertEqual(loaded["labels"]["img1.jpg"][0]["box"], [10, 10, 50, 60])

        self.assertEqual(len(loaded["refs"]["cup"]), 1)
        self.assertEqual(loaded["refs"]["cup"][0].size, (40, 40))

        self.assertEqual(len(loaded["pending"]["img1.jpg"]), 1)
        self.assertEqual(loaded["pending"]["img1.jpg"][0]["class"], "bottle")

        self.assertEqual(loaded["times"], times)
        self.assertEqual(loaded["rejections"], rejections)
        self.assertEqual(loaded["settings"]["train_pct"], 75)

    def test_backup_zip_export_and_restore(self):
        img = create_sample_image(120, 80, (100, 200, 150))
        crop = create_sample_image(30, 30, (200, 100, 50))
        images = {"photo.jpg": img}
        labels = {"photo.jpg": [{"class": "vehicle", "box": [5, 5, 50, 50], "source": "manual"}]}
        refs = {"vehicle": [crop]}
        pending = {}
        times = [{"seconds": 1.2, "images": 1}]
        rejections = [1]
        settings = {"seed_val": 99}

        zip_bytes = export_backup_zip(
            images=images,
            labels=labels,
            refs=refs,
            pending=pending,
            times=times,
            rejections=rejections,
            settings=settings,
        )

        self.assertIsInstance(zip_bytes, bytes)
        self.assertGreater(len(zip_bytes), 100)

        # Test restoring
        restored = restore_backup_zip(zip_bytes)
        self.assertIn("photo.jpg", restored["images"])
        self.assertEqual(restored["labels"]["photo.jpg"][0]["class"], "vehicle")
        self.assertEqual(len(restored["refs"]["vehicle"]), 1)
        self.assertEqual(restored["settings"]["seed_val"], 99)

    def test_malformed_backup_rejected_without_overwriting(self):
        ws_dir = self.tmp_path / "protected_ws"
        img = create_sample_image(50, 50)
        save_workspace(
            ws_dir,
            images={"valid.png": img},
            labels={},
            refs={},
            pending={},
            times=[],
            rejections=[],
            settings={"status": "initial_valid"},
        )

        # 1. Non-zip data
        with self.assertRaises(WorkspaceValidationError) as ctx:
            restore_backup_zip(b"this is not a zip file")
        self.assertIn("not a valid ZIP", str(ctx.exception))

        # Verify original save is untouched
        loaded = load_workspace(ws_dir)
        self.assertIn("valid.png", loaded["images"])

        # 2. Corrupted JSON inside ZIP
        bad_zip_buf = io.BytesIO()
        with zipfile.ZipFile(bad_zip_buf, "w") as zf:
            zf.writestr("project.json", "{corrupted json...")
        with self.assertRaises(WorkspaceValidationError):
            restore_backup_zip(bad_zip_buf.getvalue())

        # 3. Path traversal attack inside ZIP
        evil_zip_buf = io.BytesIO()
        with zipfile.ZipFile(evil_zip_buf, "w") as zf:
            zf.writestr("../evil.txt", "exploit")
            zf.writestr("project.json", json.dumps({"format": "labelforge-workspace", "version": 1, "images": []}))
        with self.assertRaises(WorkspaceValidationError) as ctx:
            restore_backup_zip(evil_zip_buf.getvalue())
        self.assertIn("Unsafe path", str(ctx.exception))

        # 4. Incompatible schema version
        incompat_zip_buf = io.BytesIO()
        with zipfile.ZipFile(incompat_zip_buf, "w") as zf:
            zf.writestr(
                "project.json",
                json.dumps({"format": "labelforge-workspace", "version": 999, "images": []}),
            )
        with self.assertRaises(WorkspaceValidationError) as ctx:
            restore_backup_zip(incompat_zip_buf.getvalue())
        self.assertIn("Unsupported schema version", str(ctx.exception))

        # Valid save remains completely intact
        loaded_after = load_workspace(ws_dir)
        self.assertEqual(loaded_after["settings"]["status"], "initial_valid")

    def test_clear_saved_workspace_does_not_affect_memory(self):
        ws_dir = self.tmp_path / "clear_ws"
        img = create_sample_image(60, 60)
        save_workspace(
            ws_dir,
            images={"test.jpg": img},
            labels={},
            refs={},
            pending={},
            times=[],
            rejections=[],
        )
        self.assertTrue(has_saved_workspace(ws_dir))

        # Clear disk
        cleared = clear_saved_workspace(ws_dir)
        self.assertTrue(cleared)
        self.assertFalse(has_saved_workspace(ws_dir))

        # In-memory dictionary remains available to the caller
        in_memory_images = {"test.jpg": img}
        self.assertIn("test.jpg", in_memory_images)

    def test_fresh_session_auto_restore_full_lifecycle(self):
        ws_dir = self.tmp_path / "lifecycle_ws"
        # 1. User uploads images, creates labels, builds reference library, has pending proposals
        img1 = create_sample_image(150, 150, (50, 100, 150))
        crop1 = create_sample_image(30, 30, (200, 100, 50))
        images = {"image_01.jpg": img1}
        labels = {"image_01.jpg": [{"class": "helmet", "box": [10, 10, 60, 60], "source": "manual"}]}
        refs = {"helmet": [crop1]}
        pending = {"image_01.jpg": [{"class": "vest", "box": [20, 70, 80, 130], "score": 0.89, "source": "auto", "id": "p_1"}]}
        times = [{"seconds": 2.1, "images": 1}]
        rejections = [1, 1]
        settings = {"model_choice": "Grounding DINO", "train_pct": 85}

        # 2. Autosave triggers
        res = save_workspace(ws_dir, images, labels, refs, pending, times, rejections, settings)
        self.assertEqual(res["status"], "Saved")

        # 3. Simulate browser refresh / server restart (session state wiped clean)
        session_images = {}
        session_labels = {}
        session_refs = {}
        session_pending = {}
        session_times = []
        session_rejections = []
        session_settings = {}

        # 4. App starts up, detects saved workspace, and restores
        self.assertTrue(has_saved_workspace(ws_dir))
        restored = load_workspace(ws_dir)
        session_images = restored["images"]
        session_labels = restored["labels"]
        session_refs = restored["refs"]
        session_pending = restored["pending"]
        session_times = restored["times"]
        session_rejections = restored["rejections"]
        session_settings = restored["settings"]

        # Verify exact recovery
        self.assertIn("image_01.jpg", session_images)
        self.assertEqual(session_images["image_01.jpg"].size, (150, 150))
        self.assertEqual(session_labels["image_01.jpg"][0]["class"], "helmet")
        self.assertEqual(session_labels["image_01.jpg"][0]["box"], [10, 10, 60, 60])
        self.assertEqual(len(session_refs["helmet"]), 1)
        self.assertEqual(session_pending["image_01.jpg"][0]["class"], "vest")
        self.assertEqual(session_pending["image_01.jpg"][0]["score"], 0.89)
        self.assertEqual(session_times[0]["seconds"], 2.1)
        self.assertEqual(len(session_rejections), 2)
        self.assertEqual(session_settings["train_pct"], 85)


if __name__ == "__main__":
    unittest.main()
