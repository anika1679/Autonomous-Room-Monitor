"""
export_map.py
 
Single file that does everything:
  1. Runs the Go2 blueprint automatically instead of having a seperate terminal
  2. Captures the map from CostMapper in real time
  3. On Ctrl-C: saves the map as .pgm / .yaml / .png
  4. Orientates it to match other maps
  5. Prints the next command to run for zone editing (work in progress)

"""
 
from __future__ import annotations
 
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.mapping.costmapper import CostMapper
 
RAW_DIR = Path("./maps/raw")
CORRECTED_DIR = Path("./maps/corrected")
_latest_map = None
 
# The following methods below aid in orientating the map (more adjustments
# to be made to make fully horizontal).
 
def _find_longest_wall_angle(img: np.ndarray) -> float | None:
    """Return angle (degrees) of the longest detected wall line, or None."""
    occupied = (img < 50).astype(np.uint8) * 255
    if occupied.sum() < 100:
        return None
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    occupied = cv2.dilate(occupied, kernel, iterations=1)
 
    lines = cv2.HoughLinesP(
        occupied,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=20,
        maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return None
 
    best_len, best_angle = -1.0, 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length > best_len:
            best_len   = length
            best_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
 
    print(f"[straighten] Longest wall: {best_len:.0f}px at {best_angle:.1f}°")
    return best_angle
 
 
def _rotate_map(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate map, expanding canvas so nothing clips. Fills new areas as unknown (205)."""
    h, w   = img.shape
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
 
    cos_a  = abs(M[0, 0])
    sin_a  = abs(M[0, 1])
    new_w  = int(h * sin_a + w * cos_a)
    new_h  = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
 
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=205,
    )
 
 
def _straighten(img: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    wall_angle = _find_longest_wall_angle(img)
 
    if wall_angle is None:
        print("[straighten] No walls detected — skipping rotation")
        return img, meta
 
    # Correction to make wall horizontal, clamped to [-45, 45]
    correction = -wall_angle
    while correction >  45: correction -= 90
    while correction < -45: correction += 90
 
    if abs(correction) < 0.5:
        print("[straighten] Already horizontal — no rotation needed")
        return img, meta
 
    print(f"[straighten] Rotating {correction:+.2f}° to align longest wall to horizontal")
    rotated = _rotate_map(img, correction)
 
    # Update origin for expanded canvas
    res = float(str(meta.get("resolution", 0.05)).strip())
    origin_raw = meta.get("origin", [0.0, 0.0, 0.0])
    if isinstance(origin_raw, list):
        ox, oy = float(origin_raw[0]), float(origin_raw[1])
    else:
        parts = str(origin_raw).strip("[]").split(",")
        ox, oy = float(parts[0]), float(parts[1])
 
    dw = (rotated.shape[1] - img.shape[1]) / 2
    dh = (rotated.shape[0] - img.shape[0]) / 2
    updated_meta = dict(meta)
    updated_meta["origin"] = [round(ox - dw * res, 6), round(oy - dh * res, 6), 0.0]
 
    return rotated, updated_meta
 
# Saves the map to the associated directories based on raw or corrected
 
def _save_map(img: np.ndarray, meta: dict, stem: str, out_dir: Path) -> dict[str, Path]:
    # Saves all the different files to their associated directories (.pgm, .png, .yaml
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
 
    pgm = out_dir / f"{stem}.pgm"
    Image.fromarray(img).save(pgm)
    paths["pgm"] = pgm
    print(f"[map] Saved {pgm}")
 
    res    = meta.get("resolution", 0.05)
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    yaml   = out_dir / f"{stem}.yaml"
    yaml.write_text(
        f"image: {stem}.pgm\n"
        f"resolution: {res}\n"
        f"origin: {origin}\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
    )
    paths["yaml"] = yaml
    print(f"[map] Saved {yaml}")
 
    rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)
    rgb[img == 205] = [180, 190, 210]   # unknown  → blue-grey
    rgb[img == 0]   = [40,  40,  40]    # occupied → dark
    rgb[img == 254] = [240, 240, 240]   # free     → light
    png = out_dir / f"{stem}.png"
    Image.fromarray(rgb).save(png)
    paths["png"] = png
    print(f"[map] Saved {png}")
 
    return paths
 
 
def _occupancy_to_image(msg) -> tuple[np.ndarray, dict]:
    """Convert dimos OccupancyGrid message to grayscale image + metadata dict."""
    arr = msg.grid   # 2D np.int8 array (height, width)
    img = np.where(arr == -1,  205,
          np.where(arr == 0,   254,
          np.where(arr >  0,     0, 205))).astype(np.uint8)
    img = np.flipud(img)   # ROS maps stored bottom-up
 
    meta = {
        "resolution": msg.resolution,
        "origin": [
            round(float(msg.origin.position.x), 6),
            round(float(msg.origin.position.y), 6),
            0.0,
        ],
    }
    return img, meta
 
 
def save_and_straighten(msg) -> None:
    """Convert, save raw map, straighten, save straight map."""
    stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 
    print(f"\n[map] {msg.width}x{msg.height} cells at {msg.resolution:.4f}m/cell")
    img, meta = _occupancy_to_image(msg)
 
    # Save raw map
    raw_paths = _save_map(img, meta, stem, RAW_DIR)
 
    # Straighten and save to corrected/
    print("[straighten] Straightening map ...")
    straight_img, straight_meta = _straighten(img, meta)
    straight_paths = _save_map(straight_img, straight_meta, stem + "_straight", CORRECTED_DIR)
 
    print(f"\n[map] Done.")
    print(f"[map] Raw        : {raw_paths['pgm']}")
    print(f"[map] Corrected  : {straight_paths['pgm']}")
    print(f"\nNext step:")
    print(f"  python map_tool.py --map {straight_paths['pgm']}")
 
 
# Main method that controls running the blueprint and collecting the data
 
def main() -> None:
    print("[setup] Building Go2 blueprint ...")
    coordinator = unitree_go2.build()
 
    cost_mapper = coordinator.get_instance(CostMapper)
    if cost_mapper is None:
        print("[warn] CostMapper not found — map will not be captured")
    else:
        print("[setup] Found CostMapper — map will be saved on Ctrl-C")
 
        def on_costmap(msg) -> None:
            global _latest_map
            _latest_map = msg
            print(
                f"\r[map] {msg.width}x{msg.height} cells  "
                f"res={msg.resolution:.4f}m  "
                f"free={msg.free_percent:.0f}%  "
                f"occupied={msg.occupied_percent:.0f}%    ",
                end="", flush=True,
            )
 
        cost_mapper.global_costmap.subscribe(on_costmap)
 
    def shutdown(sig, frame) -> None:
        print("\n\n[shutdown] Ctrl-C received — saving map ...")
        if _latest_map is not None:
            save_and_straighten(_latest_map)
        else:
            print("[shutdown] No map received — robot may not have moved enough.")
        sys.exit(0)
 
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)
 
    print("[setup] Running. Navigate normally. Press Ctrl-C when done.\n")
 
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)
 
 
if __name__ == "__main__":
    main()
