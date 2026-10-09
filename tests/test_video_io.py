"""Threaded frame-reader regressions; no model weights or GPU required."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from video_io import FrameReader


class Capture:
    def __init__(self, frames, failure=None):
        self.frames = iter(frames)
        self.failure = failure
        self.released = False
        self.reads = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FRAME_COUNT: 3}.get(prop, 0)

    def read(self):
        self.reads += 1
        try:
            return True, next(self.frames)
        except StopIteration:
            if self.failure is not None:
                raise self.failure
            return False, None

    def release(self):
        self.released = True


def image():
    return np.zeros((48, 64, 3), dtype=np.uint8)


class FrameReaderTests(unittest.TestCase):
    def test_capture_failure_reaches_consumer_after_buffered_frames(self):
        first = image()
        failure = RuntimeError("decoder failed")
        capture = Capture([first], failure)
        with patch("video_io.cv2.VideoCapture", return_value=capture), patch("threading.excepthook"):
            with FrameReader("fixture.mp4", buffer=1) as reader:
                iterator = iter(reader)
                self.assertIs(next(iterator), first)
                with self.assertRaises(RuntimeError) as caught:
                    next(iterator)
                self.assertIs(caught.exception, failure)
            self.assertFalse(reader._thread.is_alive())
        self.assertTrue(capture.released)

    def test_resize_failure_is_not_reported_as_clean_eof(self):
        capture = Capture([image()])
        failure = ValueError("resize failed")
        with patch("video_io.cv2.VideoCapture", return_value=capture), \
                patch("video_io.resize", side_effect=failure), patch("threading.excepthook"):
            with FrameReader("fixture.mp4", width=32) as reader:
                with self.assertRaises(ValueError) as caught:
                    list(reader)
                self.assertIs(caught.exception, failure)
            self.assertFalse(reader._thread.is_alive())
        self.assertTrue(capture.released)

    def test_clean_eof_preserves_frame_order(self):
        frames = [image(), image(), image()]
        capture = Capture(frames)
        with patch("video_io.cv2.VideoCapture", return_value=capture):
            with FrameReader("fixture.mp4", buffer=1) as reader:
                actual = list(reader)
        self.assertEqual(len(actual), 3)
        self.assertTrue(all(a is b for a, b in zip(actual, frames)))
        self.assertTrue(capture.released)

    def test_max_frames_stops_without_reading_another_frame(self):
        capture = Capture([image(), image(), image()], RuntimeError("should not read beyond cap"))
        with patch("video_io.cv2.VideoCapture", return_value=capture):
            with FrameReader("fixture.mp4", max_frames=2) as reader:
                self.assertEqual(len(list(reader)), 2)
        self.assertEqual(capture.reads, 2)
        self.assertTrue(capture.released)

    def test_early_consumer_exit_stops_a_producer_with_small_buffer(self):
        capture = Capture([image() for _ in range(20)])
        with patch("video_io.cv2.VideoCapture", return_value=capture):
            with FrameReader("fixture.mp4", buffer=1) as reader:
                next(iter(reader))
            self.assertFalse(reader._thread.is_alive())
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
