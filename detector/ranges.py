from collections import namedtuple, defaultdict

TimeRange = namedtuple("TimeRange", ["start", "end", "match_type", "label", "best_confidence"])


def merge_into_ranges(detections, gap_tolerance_sec=3.0):
    grouped = defaultdict(list)
    for detection in detections:
        grouped[(detection.match_type, detection.label)].append(detection)

    ranges = []
    for (match_type, label), group in grouped.items():
        group.sort(key=lambda d: d.frame_time_sec)

        current_start = group[0].frame_time_sec
        current_end = group[0].frame_time_sec
        current_best = group[0].confidence

        for detection in group[1:]:
            if detection.frame_time_sec - current_end <= gap_tolerance_sec:
                current_end = detection.frame_time_sec
                current_best = max(current_best, detection.confidence)
            else:
                ranges.append(TimeRange(current_start, current_end, match_type, label, current_best))
                current_start = detection.frame_time_sec
                current_end = detection.frame_time_sec
                current_best = detection.confidence

        ranges.append(TimeRange(current_start, current_end, match_type, label, current_best))

    ranges.sort(key=lambda r: r.start)
    return ranges
