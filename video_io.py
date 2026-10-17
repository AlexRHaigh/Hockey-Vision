"""Read video frames on a background thread, so decoding the next frames overlaps with processing
the current one (OpenCV releases the GIL while it decodes). On the Jetson's CPU, decoding 1080p
is a noticeable share of each frame's time.

    with FrameReader("game.mp4") as video:
        for frame in video:
            ...

With `width`, every frame is resized to that width (keeping the aspect ratio), e.g. to test how the
models do on lower-resolution video. width / height are then the resized size; source_width /
source_height stay the video's own.

Decoding uses the hardware decoder when there is one (`decoder="auto"`): on a Jetson whose OpenCV
was built with GStreamer (JetPack's own python3-opencv), NVDEC through nvv4l2decoder; elsewhere
FFmpeg's hardware acceleration, when OpenCV's FFmpeg has any. Otherwise, or with `decoder="cpu"`,
FFmpeg on the CPU. FrameReader.decoder says which one is in use.

Also here: CutDetector (camera cuts from colour histograms, computed while the frames are already
decoded) and BackgroundWriter (runs output writing on a thread, off the inference loop).
"""

import queue
import threading
from pathlib import Path

import cv2


def resized_size(width, height, new_width):
    """(w, h) for a frame resized to new_width keeping its aspect ratio (h rounded to even, as video
    encoders want), or None to keep the frame as it is."""
    if not new_width or new_width == width:
        return None
    return new_width, max(2, round(height * new_width / width / 2) * 2)


def resize(frame, size):
    if size is None:
        return frame
    shrink = size[0] < frame.shape[1]
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)


CUT_CORRELATION = 0.7  # consecutive frames whose colour histograms correlate less than this are a camera cut


class CutDetector:
    """Spots camera cuts: a frame whose hue/saturation histogram correlates poorly with the last one's."""

    def __init__(self):
        self.prev = None

    def update(self, frame):
        """True when `frame` starts a new shot."""
        h, w = frame.shape[:2]
        # The histogram of a quarter-size frame is as good for spotting cuts, and much cheaper.
        small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        cut = self.prev is not None and cv2.compareHist(self.prev, hist, cv2.HISTCMP_CORREL) < CUT_CORRELATION
        self.prev = hist
        return bool(cut)


def video_info(path):
    """(fps, width, height, frame count) from the container, without decoding any frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open {path}")
    info = (cap.get(cv2.CAP_PROP_FPS) or 30, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def _jetson_gstreamer():
    """True on a Jetson whose OpenCV can run GStreamer pipelines."""
    if not Path("/etc/nv_tegra_release").exists():
        return False
    return any(line.strip().startswith("GStreamer:") and "YES" in line
               for line in cv2.getBuildInformation().splitlines())


def _nvdec_pipeline(path):
    # parsebin demuxes and parses whatever the container holds (H.264 / H.265 ...) for NVDEC;
    # nvvidconv copies the frame out of GPU memory, videoconvert drops the alpha for BGR.
    location = str(Path(path).resolve()).replace('"', '\\"')
    return (f'filesrc location="{location}" ! parsebin ! nvv4l2decoder ! nvvidconv ! '
            'video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false')


def open_capture(path, decoder="auto"):
    """(cv2.VideoCapture, first frame or None, decoder name). The first frame has already been read
    from the capture (to check a hardware decoder really works), so it must be used first."""
    if decoder == "auto" and _jetson_gstreamer():
        cap = cv2.VideoCapture(_nvdec_pipeline(path), cv2.CAP_GSTREAMER)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            return cap, frame, "nvdec (GStreamer)"
        cap.release()
    if decoder == "auto":
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        if cap.isOpened() and cap.get(cv2.CAP_PROP_HW_ACCELERATION) > 0:
            return cap, None, "ffmpeg (hardware)"
        cap.release()
    cap = cv2.VideoCapture(str(path))
    return cap, None, "ffmpeg (cpu)"


class FrameReader:
    def __init__(self, path, max_frames=0, buffer=8, width=None, decoder="auto"):
        # The container's metadata is read the plain way: a GStreamer pipeline doesn't report it.
        self.fps, self.source_width, self.source_height, self.total = video_info(path)
        self.cap, self._first, self.decoder = open_capture(path, decoder)
        if not self.cap.isOpened():
            raise OSError(f"Could not open {path}")
        self.size = resized_size(self.source_width, self.source_height, width)
        self.width, self.height = self.size or (self.source_width, self.source_height)
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
                if self._first is not None:
                    ok, frame, self._first = True, self._first, None
                else:
                    ok, frame = self.cap.read()
                if not ok:
                    break
                self._put(resize(frame, self.size))
                n += 1
        except Exception as exc:
            self._put(exc)
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
            if isinstance(frame, Exception):
                raise frame
            yield frame

    def close(self):
        self._stop.set()
        self._thread.join()
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BackgroundWriter:
    """Runs jobs (drawing, encoding, JSON writing) in order on one background thread, so the
    inference loop can start the next frame meanwhile. At most `buffer` jobs wait; submit() blocks
    beyond that. An exception in a job is raised again by the next submit() or by close()."""

    def __init__(self, buffer=8):
        self._queue = queue.Queue(maxsize=buffer)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while (job := self._queue.get()) is not None:
            if self._error is None:
                try:
                    job()
                except BaseException as e:  # handed to the main thread
                    self._error = e

    def _raise(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def submit(self, job):
        self._raise()
        self._queue.put(job)

    def close(self):
        """Wait for every submitted job to finish."""
        self._queue.put(None)
        self._thread.join()
        self._raise()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
