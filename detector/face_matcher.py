import os

import cv2

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_DETECTOR_MODEL = os.path.join(_MODELS_DIR, "face_detection_yunet_2023mar.onnx")
_RECOGNIZER_MODEL = os.path.join(_MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# SFace's published verification threshold for cosine similarity — pairs of
# the same person typically score above this on standard benchmarks.
_DEFAULT_TOLERANCE = 0.363


class FaceMatcherError(Exception):
    pass


class FaceMatcher:
    """OpenCV DNN face matcher: YuNet for detection, SFace for recognition.
    Same `.match(frame_bgr) -> float | None` shape as LogoMatcher so
    scanner.py treats both uniformly. Both models are bundled ONNX files
    under detector/models/ — no extra dependency beyond opencv-python(-headless),
    and no dlib/cmake compile step required.
    """

    def __init__(self, face_path, label, tolerance=_DEFAULT_TOLERANCE):
        self.label = label
        self.tolerance = tolerance
        self._face_path = face_path

        image = cv2.imread(face_path)
        if image is None:
            raise FaceMatcherError(f"Could not read face reference image: {face_path}")

        self._detector = cv2.FaceDetectorYN_create(
            _DETECTOR_MODEL, "", (image.shape[1], image.shape[0])
        )
        _, faces = self._detector.detect(image)
        if faces is None or len(faces) == 0:
            raise FaceMatcherError(
                f"No face found in reference image for '{label}'. "
                f"Use a clear, front-facing photo with a single visible face."
            )

        best_face = max(faces, key=lambda f: f[-1])

        self._recognizer = cv2.FaceRecognizerSF_create(_RECOGNIZER_MODEL, "")
        aligned = self._recognizer.alignCrop(image, best_face)
        self._reference_feature = self._recognizer.feature(aligned)

    def match(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        best_score = None
        for face in faces:
            aligned = self._recognizer.alignCrop(frame_bgr, face)
            feature = self._recognizer.feature(aligned)
            score = self._recognizer.match(
                self._reference_feature, feature, cv2.FaceRecognizerSF_FR_COSINE
            )
            if best_score is None or score > best_score:
                best_score = score

        if best_score is not None and best_score >= self.tolerance:
            return float(min(1.0, best_score))
        return None

    def __reduce__(self):
        # cv2 DNN net objects aren't picklable — reconstruct from the source
        # image instead, so instances can cross process boundaries for
        # multiprocessing-based scanning (see scanner.py).
        return (self.__class__, (self._face_path, self.label, self.tolerance))
