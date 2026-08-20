import subprocess


class ClipperError(Exception):
    pass


def cut_clip(video_path, start, end, out_path, pad=2.0):
    clip_start = max(0.0, start - pad)
    clip_end = end + pad

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-to", str(clip_end),
        "-i", video_path,
        "-c", "copy",
        out_path,
    ]

    # ffmpeg's output isn't guaranteed to be valid in the platform's default
    # encoding (e.g. cp1252 on Windows) — decode as UTF-8 and substitute
    # anything invalid instead of letting the decode crash outright.
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise ClipperError(
            f"ffmpeg failed to cut clip ({clip_start:.1f}s - {clip_end:.1f}s): {result.stderr[-500:]}"
        )

    return out_path
