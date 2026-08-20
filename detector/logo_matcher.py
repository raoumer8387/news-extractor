import cv2
import numpy as np


class LogoMatcherError(Exception):
    pass


class LogoMatcher:
    """SIFT feature matching + Lowe's ratio test + RANSAC homography
    verification. More robust than plain ORB distance-threshold matching
    against motion blur, scale/rotation differences, and background clutter
    in real broadcast footage — background keypoints rarely agree on a
    consistent geometric transform, so homography inliers are a much cleaner
    signal than raw match count. Still classical CV, no training data
    required. Swappable for a YOLO-trained detector later via the same
    `.match(frame_gray) -> float | None` interface.
    """

    RATIO_TEST_THRESHOLD = 0.75

    def __init__(self, logo_path, label, min_good_matches=12):
        self.label = label
        self.min_good_matches = min_good_matches
        self._logo_path = logo_path

        image = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise LogoMatcherError(f"Could not read logo image: {logo_path}")

        self._sift = cv2.SIFT_create()
        self._matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),  # FLANN_INDEX_KDTREE
            dict(checks=50),
        )

        self._ref_keypoints, self._ref_descriptors = self._sift.detectAndCompute(image, None)
        if self._ref_descriptors is None or len(self._ref_keypoints) < min_good_matches:
            raise LogoMatcherError(
                f"Logo image '{label}' has too few detectable features "
                f"({0 if self._ref_descriptors is None else len(self._ref_keypoints)} found, "
                f"need at least {min_good_matches}). Try a higher-resolution or higher-contrast image."
            )

    def match(self, frame_gray):
        keypoints, descriptors = self._sift.detectAndCompute(frame_gray, None)
        if descriptors is None or len(keypoints) < 2:
            return None

        pairs = self._matcher.knnMatch(self._ref_descriptors, descriptors, k=2)

        good = []
        for pair in pairs:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.RATIO_TEST_THRESHOLD * n.distance:
                good.append(m)

        if len(good) < self.min_good_matches:
            return None

        src_pts = np.float32([self._ref_keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < self.min_good_matches:
            return None

        return min(1.0, inliers / 20.0)

    def __reduce__(self):
        # cv2 SIFT/FlannBasedMatcher aren't picklable — reconstruct from the
        # source image instead, so instances can cross process boundaries
        # for multiprocessing-based scanning (see scanner.py).
        return (self.__class__, (self._logo_path, self.label, self.min_good_matches))
