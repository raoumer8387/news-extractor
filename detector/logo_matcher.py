import cv2


class LogoMatcherError(Exception):
    pass


class LogoMatcher:
    """ORB-based logo matcher. Swappable: a future YOLO-based matcher only
    needs to implement the same `.match(frame_gray) -> float | None` interface
    so scanner.py doesn't need to change.
    """

    def __init__(self, logo_path, label, min_good_matches=12):
        self.label = label
        self.min_good_matches = min_good_matches
        self._logo_path = logo_path

        image = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise LogoMatcherError(f"Could not read logo image: {logo_path}")

        self._orb = cv2.ORB_create()
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self._ref_keypoints, self._ref_descriptors = self._orb.detectAndCompute(image, None)
        if self._ref_descriptors is None or len(self._ref_keypoints) < min_good_matches:
            raise LogoMatcherError(
                f"Logo image '{label}' has too few detectable features "
                f"({0 if self._ref_descriptors is None else len(self._ref_keypoints)} found, "
                f"need at least {min_good_matches}). Try a higher-resolution or higher-contrast image."
            )

    def match(self, frame_gray):
        keypoints, descriptors = self._orb.detectAndCompute(frame_gray, None)
        if descriptors is None or len(keypoints) == 0:
            return None

        matches = self._bf.match(self._ref_descriptors, descriptors)
        good = [m for m in matches if m.distance < 60]

        if len(good) >= self.min_good_matches:
            return min(1.0, len(good) / 40.0)
        return None

    def __reduce__(self):
        # cv2.ORB/BFMatcher aren't picklable — reconstruct from the source
        # image instead, so instances can cross process boundaries for
        # multiprocessing-based scanning (see scanner.py).
        return (self.__class__, (self._logo_path, self.label, self.min_good_matches))
