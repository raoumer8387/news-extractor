import os
import queue as queue_module
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import Manager

import cv2

Detection = namedtuple("Detection", ["frame_time_sec", "match_type", "label", "confidence"])

# Below this many sampled frames per worker, process-spawning overhead (each
# worker re-imports cv2/dlib and re-loads reference images) isn't worth it.
MIN_SAMPLES_PER_WORKER = 20
MAX_WORKERS = 8


class ScannerError(Exception):
    pass


def _probe_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ScannerError(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return native_fps, total_frames


def _run_matchers(frame, logo_matchers, face_matchers, current_sec):
    detections = []
    frame_gray = None

    for matcher in logo_matchers:
        if frame_gray is None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        confidence = matcher.match(frame_gray)
        if confidence is not None:
            detections.append(Detection(current_sec, "logo", matcher.label, confidence))

    for matcher in face_matchers:
        # YuNet/SFace (unlike the previous dlib pipeline) work directly on
        # OpenCV's native BGR frames — no color conversion needed.
        confidence = matcher.match(frame)
        if confidence is not None:
            detections.append(Detection(current_sec, "face", matcher.label, confidence))

    return detections


def _scan_frame_range(
    video_path, logo_matchers, face_matchers, frame_interval, native_fps,
    start_frame, end_frame, progress_callback=None, progress_queue=None,
):
    """Scans frames [start_frame, end_frame). Only every frame_interval-th
    frame is fully decoded (via .read()) and passed to the matchers; frames
    in between are skipped with .grab(), which avoids the color-conversion /
    buffer-copy cost of .read() for frames we're going to discard anyway.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ScannerError(f"Could not open video: {video_path}")

    detections = []
    frame_idx = start_frame

    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while frame_idx < end_frame:
            is_sample = (frame_idx % frame_interval == 0)

            if is_sample:
                ret, frame = cap.read()
            else:
                ret = cap.grab()
                frame = None

            if not ret:
                break

            if is_sample:
                current_sec = frame_idx / native_fps
                detections.extend(_run_matchers(frame, logo_matchers, face_matchers, current_sec))

                if progress_callback is not None:
                    progress_callback(current_sec)
                if progress_queue is not None:
                    progress_queue.put(current_sec)

            frame_idx += 1
    finally:
        cap.release()

    return detections


def scan_video(video_path, logo_matchers, face_matchers, sample_fps=1.0, progress_callback=None):
    native_fps, total_frames = _probe_video(video_path)
    total_sec = total_frames / native_fps if native_fps > 0 else 0
    frame_interval = max(1, round(native_fps / sample_fps))
    num_sample_points = total_frames // frame_interval + 1

    num_workers = max(1, min(
        os.cpu_count() or 1, MAX_WORKERS, num_sample_points // MIN_SAMPLES_PER_WORKER,
    ))

    if num_workers <= 1:
        def on_sample(current_sec):
            if progress_callback is not None:
                progress_callback(current_sec, total_sec)

        return _scan_frame_range(
            video_path, logo_matchers, face_matchers, frame_interval, native_fps,
            0, total_frames, progress_callback=on_sample,
        )

    samples_per_chunk = -(-num_sample_points // num_workers)  # ceil
    chunks = []
    for i in range(num_workers):
        start_sample = i * samples_per_chunk
        end_sample = min((i + 1) * samples_per_chunk, num_sample_points)
        if start_sample >= end_sample:
            break
        chunks.append((
            start_sample * frame_interval,
            min(end_sample * frame_interval, total_frames),
        ))

    processed_samples = 0
    detections = []

    with Manager() as manager:
        progress_queue = manager.Queue()

        with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _scan_frame_range, video_path, logo_matchers, face_matchers,
                    frame_interval, native_fps, start_frame, end_frame,
                    None, progress_queue,
                )
                for start_frame, end_frame in chunks
            ]

            pending = set(futures)
            while pending:
                _, pending = wait(pending, timeout=0.2)
                while True:
                    try:
                        progress_queue.get_nowait()
                    except queue_module.Empty:
                        break
                    processed_samples += 1
                    if progress_callback is not None:
                        # Chunks scan disjoint time ranges concurrently, so
                        # there's no single "current playhead position"
                        # anymore — this is an overall-progress fraction
                        # mapped onto the same (elapsed, total) shape the
                        # callback already expects.
                        fraction = min(1.0, processed_samples / num_sample_points)
                        progress_callback(fraction * total_sec, total_sec)

            for future in futures:
                detections.extend(future.result())

    return detections
