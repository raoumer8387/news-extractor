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

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ClipperError(
            f"ffmpeg failed to cut clip ({clip_start:.1f}s - {clip_end:.1f}s): {result.stderr[-500:]}"
        )

    return out_path
