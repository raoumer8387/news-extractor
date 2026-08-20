import os
import uuid

import streamlit as st

from detector import scan_video, merge_into_ranges, cut_clip
from detector.logo_matcher import LogoMatcher, LogoMatcherError
from detector.face_matcher import FaceMatcher, FaceMatcherError
from detector.scanner import ScannerError
from detector.clipper import ClipperError

UPLOADS_DIR = "uploads"
CLIPS_DIR = os.path.join("outputs", "clips")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

st.set_page_config(page_title="BrandWatch", layout="wide")


def format_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def format_duration(seconds):
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def save_uploaded_file(uploaded_file, subdir=""):
    target_dir = os.path.join(UPLOADS_DIR, subdir) if subdir else UPLOADS_DIR
    os.makedirs(target_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
    path = os.path.join(target_dir, unique_name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


if "results" not in st.session_state:
    st.session_state.results = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

st.title("📺 BrandWatch")
st.caption("Scan a news broadcast for logo and face appearances — get timestamps and downloadable clips.")

with st.sidebar:
    st.header("1. Upload video")
    video_file = st.file_uploader("News broadcast recording", type=["mp4", "mkv", "mov"])

    st.header("2. Reference images")
    logo_files = st.file_uploader(
        "Logo image(s)", type=["png", "jpg", "jpeg", "jfif"], accept_multiple_files=True
    )
    face_files = st.file_uploader(
        "Face image(s)", type=["png", "jpg", "jpeg", "jfif"], accept_multiple_files=True
    )

    st.header("3. Settings")
    sample_fps = st.slider("Sample rate (frames/sec scanned)", 0.5, 3.0, 1.0, step=0.5)
    gap_tolerance = st.slider("Gap tolerance (sec)", 0, 15, 3, step=1)
    make_clips = st.checkbox("Cut clips for each detection", value=True)

    run_scan = st.button("Run scan", type="primary", use_container_width=True)

if run_scan:
    if not video_file:
        st.error("Please upload a news video first.")
    elif not logo_files and not face_files:
        st.error("Please upload at least one logo or face reference image.")
    else:
        video_path = save_uploaded_file(video_file)
        st.session_state.video_path = video_path

        logo_matchers = []
        face_matchers = []
        setup_failed = False

        for f in logo_files or []:
            ref_path = save_uploaded_file(f, subdir="refs")
            try:
                logo_matchers.append(LogoMatcher(ref_path, label=os.path.splitext(f.name)[0]))
            except LogoMatcherError as e:
                st.error(str(e))
                setup_failed = True

        for f in face_files or []:
            ref_path = save_uploaded_file(f, subdir="refs")
            try:
                face_matchers.append(FaceMatcher(ref_path, label=os.path.splitext(f.name)[0]))
            except FaceMatcherError as e:
                st.error(str(e))
                setup_failed = True

        if not setup_failed:
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(current_sec, total_sec):
                fraction = min(1.0, current_sec / total_sec) if total_sec > 0 else 0
                progress_bar.progress(fraction)
                status_text.text(f"Scanning {format_time(current_sec)} / {format_time(total_sec)}")

            try:
                with st.spinner("Scanning video..."):
                    detections = scan_video(
                        video_path, logo_matchers, face_matchers,
                        sample_fps=sample_fps, progress_callback=progress_callback,
                    )
                progress_bar.progress(1.0)
                status_text.text("Scan complete.")

                ranges = merge_into_ranges(detections, gap_tolerance_sec=gap_tolerance)

                rows = []
                for r in ranges:
                    clip_path = None
                    if make_clips:
                        clip_name = f"{r.match_type}_{r.label}_{int(r.start)}_{int(r.end)}_{uuid.uuid4().hex[:6]}.mp4"
                        clip_path = os.path.join(CLIPS_DIR, clip_name)
                        try:
                            cut_clip(video_path, r.start, r.end, clip_path)
                        except ClipperError as e:
                            st.warning(f"Could not cut clip for {r.label} at {format_time(r.start)}: {e}")
                            clip_path = None

                    rows.append({
                        "type": r.match_type,
                        "label": r.label,
                        "start": r.start,
                        "end": r.end,
                        "confidence": r.best_confidence,
                        "clip_path": clip_path,
                    })

                st.session_state.results = rows

            except ScannerError as e:
                st.error(str(e))

if st.session_state.results is not None:
    rows = st.session_state.results

    st.header("Results")

    if not rows:
        st.info("No logo or face matches were found in this video. Try lowering the sample rate's "
                 "granularity, increasing gap tolerance, or using a clearer reference image.")
    else:
        summary = {}
        for r in rows:
            key = (r["type"], r["label"])
            duration = r["end"] - r["start"]
            if key not in summary:
                summary[key] = {"count": 0, "total_duration": 0.0}
            summary[key]["count"] += 1
            summary[key]["total_duration"] += duration

        st.subheader("Summary")
        for (match_type, label), stats in summary.items():
            kind = "logo" if match_type == "logo" else "face"
            st.markdown(
                f"- **{label}** ({kind}) appeared **{stats['count']}** time"
                f"{'s' if stats['count'] != 1 else ''}, "
                f"total **{format_duration(stats['total_duration'])}** on screen"
            )

        st.subheader("Detections")
        table_rows = sorted(rows, key=lambda r: r["start"])
        table_data = [
            {
                "Type": "Logo" if r["type"] == "logo" else "Face",
                "Reference": r["label"],
                "Start": format_time(r["start"]),
                "End": format_time(r["end"]),
                "Duration": format_duration(r["end"] - r["start"]),
                "Confidence": f"{r['confidence']:.0%}",
            }
            for r in table_rows
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        st.subheader("Clips")
        for i, r in enumerate(table_rows):
            if not r["clip_path"] or not os.path.exists(r["clip_path"]):
                continue
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{r['label']}** ({r['type']}) — "
                    f"{format_time(r['start'])} to {format_time(r['end'])} "
                    f"(confidence {r['confidence']:.0%})"
                )
                st.video(r["clip_path"])
            with col2:
                with open(r["clip_path"], "rb") as f:
                    st.download_button(
                        "Download clip",
                        data=f.read(),
                        file_name=os.path.basename(r["clip_path"]),
                        mime="video/mp4",
                        key=f"download_{i}",
                    )
