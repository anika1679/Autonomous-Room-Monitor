"""
Predefined zones for associating detected objects with named regions and colors.
Zones can be rectangles (x1, y1, x2, y2) or polygons (list of (x, y) points).
Coordinates are in pixels; use 0–1 fractions for relative layout (see get_zones_for_frame).
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class Zone:
    """A zone with a name, color (BGR), and contour for point-in-zone tests."""
    name: str
    color: Tuple[int, int, int]  # BGR
    contour: np.ndarray  # shape (N, 1, 2), dtype int32


def point_in_zone(px: float, py: float, zone: Zone) -> bool:
    """Return True if (px, py) is inside the zone contour."""
    result = cv2.pointPolygonTest(zone.contour, (px, py), False)
    return result >= 0


def get_zone_for_point(px: float, py: float, zones: List[Zone]) -> Optional[Zone]:
    """Return the first zone containing (px, py), or None."""
    for zone in zones:
        if point_in_zone(px, py, zone):
            return zone
    return None


# Default zones defined as fractions of frame (0–1). Convert to pixels in get_zones_for_frame.
# Each: (name, BGR color, list of (x_frac, y_frac) forming a polygon; last point auto-closed).
DEFAULT_ZONE_SPECS: List[Tuple[str, Tuple[int, int, int], List[Tuple[float, float]]]] = [
    ("Zone A (Left)", (80, 180, 255), [(0.0, 0.0), (0.33, 0.0), (0.33, 1.0), (0.0, 1.0)]),   # Orange
    ("Zone B (Center)", (80, 255, 80), [(0.33, 0.0), (0.66, 0.0), (0.66, 1.0), (0.33, 1.0)]), # Green
    ("Zone C (Right)", (255, 80, 80), [(0.66, 0.0), (1.0, 0.0), (1.0, 1.0), (0.66, 1.0)]),   # Blue
]


def get_zones_for_frame(
    width: int,
    height: int,
    specs: Optional[List[Tuple[str, Tuple[int, int, int], List[Tuple[float, float]]]]] = None,
) -> List[Zone]:
    """Build Zone list from frame dimensions. Uses DEFAULT_ZONE_SPECS if specs is None."""
    if specs is None:
        specs = DEFAULT_ZONE_SPECS
    zones: List[Zone] = []
    for name, color, points_frac in specs:
        pts = np.array(
            [[int(x * width), int(y * height)] for x, y in points_frac],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        zones.append(Zone(name=name, color=color, contour=pts))
    return zones
