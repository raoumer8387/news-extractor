"""Standalone smoke test for the detector modules — run before building the UI.

Usage:
    python test_detectors.py <video_path> [--logo logo.png] [--face face.jpg]
"""
import argparse
import sys

from detector import LogoMatcher, FaceMatcher, scan_video, merge_into_ranges
from detector.logo_matcher import LogoMatcherError
from detector.face_matcher import FaceMatcherError
from detector.scanner import ScannerError


def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--logo", action="append", default=[])
    parser.add_argument("--face", action="append", default=[])
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--gap-tolerance", type=float, default=3.0)
    args = parser.parse_args()

    logo_matchers = []
    for path in args.logo:
        try:
            logo_matchers.append(LogoMatcher(path, label=path))
            print(f"Loaded logo matcher: {path}")
        except LogoMatcherError as e:
            print(f"ERROR loading logo '{path}': {e}", file=sys.stderr)
            sys.exit(1)

    face_matchers = []
    for path in args.face:
        try:
            face_matchers.append(FaceMatcher(path, label=path))
            print(f"Loaded face matcher: {path}")
        except FaceMatcherError as e:
            print(f"ERROR loading face '{path}': {e}", file=sys.stderr)
            sys.exit(1)

    if not logo_matchers and not face_matchers:
        print("Provide at least one --logo or --face reference image.", file=sys.stderr)
        sys.exit(1)

    def progress_callback(current_sec, total_sec):
        print(f"\rScanning {format_time(current_sec)} / {format_time(total_sec)}", end="", flush=True)

    try:
        detections = scan_video(
            args.video_path, logo_matchers, face_matchers,
            sample_fps=args.sample_fps, progress_callback=progress_callback,
        )
    except ScannerError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n\n{len(detections)} raw detections")

    ranges = merge_into_ranges(detections, gap_tolerance_sec=args.gap_tolerance)
    print(f"{len(ranges)} merged ranges:\n")
    for r in ranges:
        print(
            f"  [{r.match_type:5s}] {r.label:20s} "
            f"{format_time(r.start)} - {format_time(r.end)} "
            f"(confidence {r.best_confidence:.2f})"
        )


if __name__ == "__main__":
    main()
