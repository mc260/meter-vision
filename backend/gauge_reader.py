"""
gauge_reader.py  —  Perspective-robust gauge reading algorithm

Keypoint convention
-------------------
  kp_id 0     Pointer base / rotation pivot (closest annotated point to axis)
  kp_id 1–2   Points along the pointer arm, outward from kp_id 0
  kp_id 3–9   Dial scale points, fixed mapping to scale values:
              3→0.0  4→1.0  5→2.0  6→3.0  7→4.0  8→5.0  9→6.0

Design note
-----------
A circle fit over kp_id 3–9 fails under perspective projection (the arc
becomes elliptical, shifting the fitted centre away from the true pivot).
Instead, kp_id 0 is used directly as the angular origin so that all
relative angles—scale points and pointer direction—are computed from the
same reference, making perspective distortion cancel out.
"""

import math
import numpy as np


SCALE_MAP: dict[int, float] = {
    3: 0.0, 4: 1.0, 5: 2.0,
    6: 3.0, 7: 4.0, 8: 5.0, 9: 6.0,
}


def _unwrap_angles(
    sorted_angle_scale: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Unwrap an arc that crosses the ±π boundary so angles are monotonic."""
    if len(sorted_angle_scale) < 2:
        return sorted_angle_scale
    angles = [a for a, _ in sorted_angle_scale]
    gaps = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[max_gap_idx] <= math.pi:
        return sorted_angle_scale
    unwrapped = [
        (a + 2 * math.pi if i <= max_gap_idx else a, v)
        for i, (a, v) in enumerate(sorted_angle_scale)
    ]
    unwrapped.sort(key=lambda t: t[0])
    return unwrapped


def _interp(x: float, xs: list[float], ys: list[float]) -> tuple[float, bool]:
    """Piecewise linear interpolation with clamping; returns (value, out_of_range)."""
    if x <= xs[0]:
        return ys[0], x < xs[0]
    if x >= xs[-1]:
        return ys[-1], x > xs[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i]), False
    return ys[-1], True


def compute_gauge_reading(
    api_response: dict,
    person_idx: int = 0,
    conf_threshold: float = 0.5,
) -> dict:
    """
    Compute a gauge reading from the API keypoint JSON.

    Returns on success:
        value, out_of_range, confidence, pointer_angle_deg, pivot, scale_points

    Returns on failure:
        {"error": <reason>}
    """
    people = api_response.get("people", [])
    if not people or person_idx >= len(people):
        return {"error": "no person detected"}

    kps: dict[int, dict] = {
        kp["kp_id"]: kp for kp in people[person_idx]["keypoints"]
    }

    # Rotation pivot = kp_id 0
    kp0 = kps.get(0)
    if kp0 is None or kp0["conf"] < conf_threshold:
        return {"error": "kp_id 0 (pointer base) missing or low confidence"}
    px, py = kp0["x"], kp0["y"]

    # Pointer direction: confidence-weighted average of kp_id 1 & 2 relative to pivot
    ptr_kps = [
        kps[i] for i in (1, 2)
        if i in kps and kps[i]["conf"] >= conf_threshold
    ]
    if not ptr_kps:
        return {"error": "kp_id 1/2 missing or low confidence"}

    dx_sum = dy_sum = w_sum = 0.0
    for kp in ptr_kps:
        w = kp["conf"]
        dx_sum += (kp["x"] - px) * w
        dy_sum += (kp["y"] - py) * w
        w_sum += w
    pointer_angle = math.atan2(dy_sum / w_sum, dx_sum / w_sum)

    # Scale point angles (all relative to the same pivot)
    scale_kps_raw = [
        (kp["x"], kp["y"], sv, kp["conf"])
        for kid, sv in SCALE_MAP.items()
        if (kp := kps.get(kid)) and kp["conf"] >= conf_threshold
    ]
    if len(scale_kps_raw) < 3:
        return {"error": f"too few scale keypoints: {len(scale_kps_raw)} (need >= 3)"}

    raw_angle_scale = sorted(
        [(math.atan2(y - py, x - px), sv) for x, y, sv, _ in scale_kps_raw],
        key=lambda t: t[0],
    )
    angle_scale = _unwrap_angles(raw_angle_scale)

    # Sync pointer angle into the same unwrapped arc
    arc_start = angle_scale[0][0]
    if pointer_angle < arc_start - math.pi:
        pointer_angle += 2 * math.pi

    value, out_of_range = _interp(
        pointer_angle,
        [a for a, _ in angle_scale],
        [s for _, s in angle_scale],
    )

    ptr_conf = sum(kp["conf"] for kp in ptr_kps) / len(ptr_kps)
    return {
        "value": round(value, 2),
        "out_of_range": out_of_range,
        "confidence": round(ptr_conf, 4),
        "pointer_angle_deg": round(math.degrees(pointer_angle), 2),
        "pivot": (round(px, 1), round(py, 1)),
        "scale_points": [(round(x, 1), round(y, 1)) for x, y, _, _ in scale_kps_raw],
    }
