"""Read video frames on a background thread, so decoding the next frames overlaps with processing
the current one (OpenCV releases the GIL while it decodes). On the Jetson's CPU, decoding 1080p
is a noticeable share of each frame's time.

    with FrameReader("game.mp4") as video:
        for frame in video:
            ...
"""

import queue
import threading

import cv2


class FrameReader:
    def __init__(self, path, max_frames=0, buffer=8):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            self.total = min(self.total, max_frames) if self.total else max_frames
        self._max_frames = max_frames
        self._queue = queue.Queue(maxsize=buffer)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        n = 0
        try:
            while not self._stop.is_set() and not (self._max_frames and n >= self._max_frames):
                ok, frame = self.cap.read()
                if not ok:
                    break
                self._put(frame)
                n += 1
        finally:
            self._put(None)  # end of video

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def __iter__(self):
        while (frame := self._queue.get()) is not None:
            yield frame

    def close(self):
        self._stop.set()
        self._thread.join()
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
