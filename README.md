# BrandWatch

MVP media monitoring tool: upload a recorded news broadcast plus reference
images of a logo and/or a person's face, and BrandWatch scans the video,
reports every time-range where they appear (with a confidence score), and
generates a downloadable clip for each detection.

Built for a client demo — the detection logic prioritizes working end-to-end
over top accuracy, but is structured so either matcher can be swapped for a
trained model later without touching `scanner.py`, `ranges.py`, or the UI.

## Prerequisites

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) installed and on your `PATH` (used to cut clips)

## Installation

```bash
pip install -r requirements.txt
```

That's it — everything (face detection/recognition included) runs on plain
`opencv-python-headless`, `numpy`, and `streamlit`. No compiled C++
dependencies, no dlib, nothing platform-specific.

## Running

```bash
streamlit run app.py
```

Then, in the browser UI:

1. Upload a news video (`.mp4`, `.mkv`, `.mov`).
2. Upload one or more reference images: a logo, a face, or both.
3. Adjust the sample rate and gap tolerance if needed (defaults are reasonable).
4. Click **Run scan** and watch the progress bar.
5. Review the results table, watch clip previews inline, and download clips.

No command line is needed after the initial setup.

Max upload size is set to 250MB in `.streamlit/config.toml` (`server.maxUploadSize`,
raised from Streamlit's 200MB default). A 1-2 hour broadcast at typical
bitrates can exceed this — raise the value further if your source files are
larger.

## How it works

- `detector/logo_matcher.py` — SIFT feature matching with Lowe's ratio test,
  verified by RANSAC homography (rejects matches whose keypoints don't agree
  on a consistent geometric transform, which filters out most background-
  clutter false positives that plain feature-distance matching lets through).
- `detector/face_matcher.py` — OpenCV's bundled DNN models: **YuNet** for face
  detection, **SFace** for the embedding used to compare against the
  reference face. Both ship as small ONNX files in `detector/models/` and run
  through `cv2.dnn` — no dlib, no compiling, and generally faster and more
  accurate on CPU than the HOG-based approach this replaced.
- `detector/scanner.py` — opens the video with OpenCV, samples frames at the
  configured rate, and runs every matcher against each sampled frame, split
  across parallel worker processes (see "Performance" below).
- `detector/ranges.py` — merges individual frame-level detections into
  continuous time ranges, tolerating small gaps (e.g. a brief occlusion).
- `detector/clipper.py` — cuts each time range (padded a couple seconds on
  either side) into its own clip via `ffmpeg`.

Both matcher classes expose the same `.match(frame) -> confidence | None`
interface, so `scanner.py` doesn't care which detection algorithm is behind
either one. To upgrade logo detection to a trained YOLO model later, write a
new class with that same interface and swap it in — no other module needs to
change.

## Testing the detectors standalone

Before touching the UI, you can sanity-check the detector pipeline directly:

```bash
python test_detectors.py path/to/video.mp4 --logo path/to/logo.png --face path/to/face.jpg
```

This prints a live progress line and the merged detection ranges to the
terminal.

## Known MVP limitations

- SIFT + homography verification handles scale/rotation/blur far better than
  raw ORB matching, but still isn't trained on the specific logo — a very
  small, heavily stylized, or badly occluded logo can still be missed. A
  YOLO-trained detector is the real fix if this matters for a specific client.
- Face matching (YuNet + SFace) is a strong general-purpose pipeline but
  isn't tuned to a specific person the way a few-shot fine-tuned model would
  be — expect occasional misses on extreme angles or very poor lighting.
- `ffmpeg -c copy` cuts on the nearest keyframe rather than an exact frame
  boundary, so clip start/end may be off by up to a couple seconds; the
  default 2s padding is meant to absorb this.
- No database — uploads and clips live on the local filesystem
  (`uploads/`, `outputs/clips/`), both gitignored.

## Performance

`detector/scanner.py` splits the video into contiguous chunks and scans them
in parallel worker processes (up to `os.cpu_count()`, capped at 8), since
per-frame detection cost is embarrassingly parallel across time ranges. For
short clips (below ~20 sampled frames per worker) it falls back to a single
process instead, since spawning workers isn't worth it for small inputs.
Within each worker, frames that aren't being sampled are skipped with
`cap.grab()` instead of `cap.read()`, avoiding the decode/color-conversion
cost for frames that would be discarded anyway.

One side effect: with multiple workers, the sidebar's "Scanning HH:MM:SS /
HH:MM:SS" text is an overall-progress approximation (samples completed so
far, scaled onto the video's duration) rather than a literal single playhead
position, since several time ranges are being scanned concurrently.

## Deploying

### Don't use Vercel

Vercel runs serverless functions (short execution limits, no persistent
process, no bundled `ffmpeg`, no support for the multiprocessing worker pool
`scanner.py` uses). It can't host a Streamlit app at all — Streamlit needs a
long-lived process with an open WebSocket connection. This isn't a
configuration problem to work around; it's the wrong kind of platform for
this app.

### Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   create a new app pointing at your repo, branch, and `app.py` as the main
   file.
3. That's it — Cloud installs `packages.txt` (just `ffmpeg`) and
   `requirements.txt` (`opencv-python-headless`, `numpy`, `streamlit`) on
   deploy. No compile step, so this should be a fast, uneventful build.

**`runtime.txt` pins Python 3.12 — don't remove it.** Without a pin,
Streamlit Cloud defaults to the newest available Python, and if that's newer
than what `numpy`/`opencv-python-headless` have published wheels for, pip
falls back to compiling from source, which can hang for an hour+ (this
happened in practice on Python 3.14 before this pin was added) and can leave
a half-installed environment where `import cv2` fails at runtime. 3.12 is
verified to have prebuilt wheels for every dependency here.

**Known constraints on the free tier** (not fully verifiable without an
actual deploy — treat as things to check, not guarantees):
- Community Cloud apps typically get on the order of 1 CPU / 1GB RAM. The
  scanner's worker count is already capped by `os.cpu_count()`, so it won't
  over-subscribe a small instance, but a 1-2 hour broadcast may still be slow
  or hit memory limits — test with a shorter clip first for the demo.
- Storage is ephemeral — uploads/clips disappear on app restart/redeploy,
  which is fine for a live demo but isn't persistent storage.
- Cloud's own upload-size handling may not fully respect
  `.streamlit/config.toml`'s `maxUploadSize` in all cases — verify with an
  actual upload once deployed, and trim the demo video if it's rejected.

If these limits turn out to be too tight for the client demo, a Docker-based
host (Render, Railway, Fly.io) gives more control over resources.

### Push to GitHub

```bash
git add .
git commit -m "Your message"
git push
```

`uploads/`, `outputs/`, and `.venv/` are gitignored, so only source, model
files, and config get pushed.
