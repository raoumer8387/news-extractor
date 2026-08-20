import cv2
import face_recognition


class FaceMatcherError(Exception):
    pass


class FaceMatcher:
    """dlib/face_recognition based face matcher. Same `.match(frame_rgb) ->
    float | None` interface as LogoMatcher so scanner.py treats both uniformly.
    """

    def __init__(self, face_path, label, tolerance=0.5):
        self.label = label
        self.tolerance = tolerance
        self._face_path = face_path

        image = face_recognition.load_image_file(face_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            raise FaceMatcherError(
                f"No face found in reference image for '{label}'. "
                f"Use a clear, front-facing photo with a single visible face."
            )

        self._reference_encoding = encodings[0]

    def match(self, frame_rgb):
        small = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
        locations = face_recognition.face_locations(small, model="hog")
        if not locations:
            return None

        encodings = face_recognition.face_encodings(small, locations)
        if not encodings:
            return None

        distances = face_recognition.face_distance(encodings, self._reference_encoding)
        best_idx = distances.argmin()
        best_distance = distances[best_idx]

        if best_distance < self.tolerance:
            return float(1 - best_distance)
        return None

    def __reduce__(self):
        # The dlib face-detector state isn't picklable — reconstruct from the
        # source image instead, so instances can cross process boundaries for
        # multiprocessing-based scanning (see scanner.py).
        return (self.__class__, (self._face_path, self.label, self.tolerance))
