# BrandWatch

MVP media monitoring tool: upload a recorded news broadcast plus reference
images of a logo and/or a person's face, and BrandWatch scans the video,
reports every time-range where they appear (with a confidence score), and
generates a downloadable clip for each detection.

Built for a client demo — the detection logic prioritizes working end-to-end
over top accuracy, but is structured so the logo matcher can be swapped for a
trained YOLO model later without touching `scanner.py`, `ranges.py`, or the UI.

## Prerequisites

- Python 3.12+ (tested on 3.13)
- [ffmpeg](https://ffmpeg.org/) installed and on your `PATH` (used to cut clips)

## Installation

**Do not run `pip install -r requirements.txt` as your first step.**
`face_recognition` normally pulls in plain `dlib`, which compiles from source
and can take 10+ minutes or fail entirely if you don't have a full C++ build
toolchain. Instead, install in this order (a script is provided):

```powershell
# Windows
.\install.ps1
```

```bash
# macOS / Linux
./install.sh
```

This runs, in order:

```
pip install dlib-bin
pip install face_recognition --no-deps
pip install git+https://github.com/ageitgey/face_recognition_models
pip install "setuptools<81"
pip install click Pillow numpy opencv-python streamlit
```

- `dlib-bin` is a prebuilt wheel that provides the same `dlib` module without
  compiling anything.
- `face_recognition` is installed with `--no-deps` so pip doesn't try to pull
  in plain `dlib` as a transitive dependency and undo the point of using
  `dlib-bin`.
- `face_recognition_models` and the remaining libraries are installed
  explicitly since `--no-deps` skipped them.
- `setuptools<81` is required because `face_recognition_models` imports
  `pkg_resources` at import time, and recent `setuptools` releases (81+) removed
  `pkg_resources` entirely. Without this pin you'll hit
  `ModuleNotFoundError: No module named 'pkg_resources'` even though
  `face_recognition_models` is installed correctly. (This is a separate pitfall
  from the `dlib` one above — discovered by actually running this install, not
  just reading the packages' docs.)

**If you "fix" this by removing `dlib-bin`/`--no-deps` and just running
`pip install -r requirements.txt`, you will likely reintroduce the slow/failing
source build of `dlib`.** `requirements.txt` is kept for reference/pinning,
not as the primary install path.

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

- `detector/logo_matcher.py` — ORB feature matching (`cv2.ORB_create` +
  `BFMatcher` with Hamming distance) between the reference logo and each
  sampled frame.
- `detector/face_matcher.py` — face detection + 128-d embedding comparison via
  `face_recognition` (dlib-based), downsizing frames 0.5x before detection for
  speed.
- `detector/scanner.py` — opens the video with OpenCV, samples frames at the
  configured rate, and runs every matcher against each sampled frame.
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

- ORB logo matching is sensitive to scale/rotation/motion blur and will miss
  small or heavily stylized logos — swap in a YOLO-trained detector for
  production accuracy.
- Face matching uses HOG (CPU-friendly, no GPU/CNN model required) — expect
  more missed detections on small or angled faces than a CNN-based model.
- `ffmpeg -c copy` cuts on the nearest keyframe rather than an exact frame
  boundary, so clip start/end may be off by up to a couple seconds; the
  default 2s padding is meant to absorb this.
- No database — uploads and clips live on the local filesystem
  (`uploads/`, `outputs/clips/`), both gitignored.

## Performance

`detector/scanner.py` splits the video into contiguous chunks and scans them
in parallel worker processes (up to `os.cpu_count()`, capped at 8), since the
dlib HOG face detector is the dominant per-frame cost and this is
embarrassingly parallel across time ranges. For short clips (below ~20
sampled frames per worker) it falls back to a single process instead, since
spawning workers (each re-imports cv2/dlib) isn't worth it for small inputs.
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
3. That's it — Cloud automatically installs `packages.txt` (system packages)
   and `requirements.txt` (Python packages) on deploy.

**Why `requirements.txt` differs from the local install scripts**: Streamlit
Cloud only supports a single `pip install -r requirements.txt` — it can't run
`install.ps1`/`install.sh`. Plain `dlib` has no prebuilt Linux wheel, so
`requirements.txt` lets it compile from source instead of using the
`dlib-bin` shortcut; `packages.txt` supplies the `cmake`/`build-essential`/
`libopenblas-dev`/`liblapack-dev` it needs to do that. This makes the first
deploy slow (~10-15 min to compile dlib once), but Cloud caches the
environment across redeploys as long as `requirements.txt`/`packages.txt`
don't change. `runtime.txt` pins Python 3.11 so this stays reproducible.

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
host (Render, Railway, Fly.io) gives more control over resources and can run
the exact `install.sh` sequence in a Dockerfile instead — ask if you want
that set up as a fallback.

### Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

`uploads/`, `outputs/`, and `.venv/` are gitignored, so only source and
config files get pushed.
