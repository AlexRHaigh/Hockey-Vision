"""Tests for the speed-ups: cut detection, the background writer, the shared downscale and INT8
engine selection. No model weights needed."""
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_models  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.engine.results import Results  # noqa: E402
from video_io import BackgroundWriter, CutDetector  # noqa: E402


def solid(bgr, h=72, w=128):
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:] = bgr
    return frame


class CutDetectorTests(unittest.TestCase):
    def test_same_shot_is_not_a_cut(self):
        cuts = CutDetector()
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (72, 128, 3), dtype=np.uint8)
        self.assertFalse(cuts.update(frame))
        self.assertFalse(cuts.update(frame))

    def test_new_colours_are_a_cut(self):
        cuts = CutDetector()
        self.assertFalse(cuts.update(solid((255, 255, 255))))  # first frame is never a cut
        self.assertTrue(cuts.update(solid((0, 0, 200))))


class BackgroundWriterTests(unittest.TestCase):
    def test_jobs_run_in_order_off_the_calling_thread(self):
        done, threads = [], set()
        with BackgroundWriter(buffer=2) as w:
            for i in range(20):
                w.submit(lambda i=i: (done.append(i), threads.add(threading.get_ident())))
        self.assertEqual(done, list(range(20)))
        self.assertNotIn(threading.get_ident(), threads)

    def test_error_is_raised_on_close(self):
        w = BackgroundWriter()

        def fail():
            raise ValueError("disk full")
        w.submit(fail)
        with self.assertRaises(ValueError):
            w.close()


class SharedDownscaleTests(unittest.TestCase):
    def test_input_size_matches_letterbox(self):
        for h, w in [(1080, 1920), (720, 1280), (1080, 1440), (481, 853), (200, 300)]:
            for imgsz in (640, [384, 640]):
                frame = np.zeros((h, w, 3), np.uint8)
                size = run_models.input_size(frame, imgsz)
                if size is None:
                    continue
                params = LetterBox(imgsz if isinstance(imgsz, list) else (imgsz, imgsz)).get_params({"img": frame})
                self.assertEqual(tuple(size), tuple(params["new_unpad"]), (h, w, imgsz))
                # The resized frame then goes through the model's letterbox without another resize.
                again = LetterBox(imgsz if isinstance(imgsz, list) else (imgsz, imgsz)).get_params(
                    {"img": np.zeros((size[1], size[0], 3), np.uint8)})
                self.assertEqual(tuple(again["new_unpad"]), tuple(size))

    def test_no_resize_for_small_frames(self):
        self.assertIsNone(run_models.input_size(np.zeros((360, 640, 3), np.uint8), 640))

    def test_boxes_are_mapped_back_to_the_frame(self):
        small = np.zeros((360, 640, 3), np.uint8)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        r = Results(small, path="", names={0: "player"},
                    boxes=torch.tensor([[10.0, 20.0, 30.0, 40.0, 7.0, 0.9, 0.0],
                                        [600.0, 300.0, 640.0, 360.0, 8.0, 0.5, 0.0]]))
        r = run_models.to_frame_coords(r, frame, (640, 360))
        self.assertEqual(r.orig_shape, (1080, 1920))
        np.testing.assert_allclose(r.boxes.xyxy.numpy(), [[30, 60, 90, 120], [1800, 900, 1920, 1080]])
        self.assertEqual(r.boxes.id.tolist(), [7.0, 8.0])  # track ids untouched
        np.testing.assert_allclose(r.boxes.conf.numpy(), [0.9, 0.5])


class ModelPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = run_models.MODELS_DIR
        run_models.MODELS_DIR = Path(self.tmp.name)

    def tearDown(self):
        run_models.MODELS_DIR = self.old
        self.tmp.cleanup()

    def touch(self, name):
        (run_models.MODELS_DIR / name).touch()

    def test_int8_engine_used_only_with_int8(self):
        for f in ("new_player_model.pt", "new_player_model.engine", "new_player_model.int8.engine"):
            self.touch(f)
        self.assertEqual(run_models.model_path("player").name, "new_player_model.engine")
        self.assertEqual(run_models.model_path("player", int8=True).name, "new_player_model.int8.engine")

    def test_puck_never_int8(self):
        for f in ("new_nano_puck.pt", "new_nano_puck.engine", "new_nano_puck.int8.engine"):
            self.touch(f)
        self.assertEqual(run_models.model_path("puck", int8=True).name, "new_nano_puck.engine")

    def test_falls_back_to_fp16_engine_then_pt(self):
        self.touch("new_rink_model.pt")
        self.assertEqual(run_models.model_path("rink", int8=True).name, "new_rink_model.pt")
        self.touch("new_rink_model.engine")
        self.assertEqual(run_models.model_path("rink", int8=True).name, "new_rink_model.engine")


if __name__ == "__main__":
    unittest.main()
