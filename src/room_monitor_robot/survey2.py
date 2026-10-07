#!/usr/bin/env python3
"""
run_survey.py

1. Boot Go2
2. Navigate to start pose
3. Let WavefrontFrontierExplorer autonomously survey for 5 minutes
4. Navigate back to start pose
5. Save map to maps/raw/ and maps/corrected/

Usage:
  export ROBOT_IP=192.168.1.xxx
  source .venv/bin/activate
  python run_survey.py --zones maps/raw/run_20260319_122835_zones.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from entropy import calculate_entropy
from dimos.navigation.frontier_exploration import WavefrontFrontierExplorer
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.core.blueprints import autoconnect
from dimos.core.introspection import to_svg
from dimos.mapping.costmapper import cost_mapper
from dimos.mapping.voxels import voxel_mapper
from dimos.navigation.frontier_exploration import wavefront_frontier_explorer
from dimos.navigation.replanning_a_star.module import replanning_a_star_planner
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import unitree_go2_basic
from dimos.mapping.costmapper import CostMapper
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

RAW_DIR        = Path("./maps/raw")
CORRECTED_DIR  = Path("./maps/corrected")
SURVEY_MINUTES = 5
_latest_map    = None

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def yaw_to_quaternion(yaw_deg: float) -> tuple[float, float, float, float]:
    yaw = math.radians(yaw_deg)
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))

def make_pose_stamped(x: float, y: float, yaw_deg: float = 0.0) -> PoseStamped:
    qx, qy, qz, qw = yaw_to_quaternion(yaw_deg)
    return PoseStamped(
        frame_id="map",
        position=[x, y, 0.0],
        orientation=[qx, qy, qz, qw],
    )

def navigate_to(planner, x: float, y: float, yaw_deg: float = 0.0,
                label: str = "", timeout: float = 60.0) -> bool:
    print(f"[nav] → {label or f'({x:.2f}, {y:.2f})'}")
    reached = threading.Event()

    def on_reached(msg) -> None:
        if getattr(msg, "data", False):
            reached.set()

    sub = planner.goal_reached.subscribe(on_reached)
    planner.goal_request.publish(make_pose_stamped(x, y, yaw_deg))

    success = reached.wait(timeout=timeout)
    try:
        sub.dispose()
    except Exception:
        pass

    if success:
        print(f"[nav] Arrived ✓")
    else:
        print(f"[nav] Timeout after {timeout:.0f}s")
    return success

# ---------------------------------------------------------------------------
# Map straightening (PCA)
# ---------------------------------------------------------------------------

def _find_dominant_wall_angle(img: np.ndarray) -> float | None:
    mapped = (img != 205).astype(np.uint8) * 255
    if mapped.sum() < 500:
        return None
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mapped, cv2.MORPH_CLOSE, kernel, iterations=3)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None
    pts = largest.reshape(-1, 2).astype(np.float32)
    _, eigenvectors = cv2.PCACompute(pts, mean=None)
    principal = eigenvectors[0]
    angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    print(f"[straighten] PCA axis: {angle_deg:.1f}°")
    correction = -angle_deg
    while correction >  45: correction -= 90
    while correction < -45: correction += 90
    print(f"[straighten] Correction: {correction:+.1f}°")
    return correction

def _rotate_map(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w   = img.shape
    cx, cy = w / 2.0, h / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w  = int(h * sin_a + w * cos_a)
    new_h  = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=205)

def _straighten(img: np.ndarray, meta: dict) -> tuple[np.ndarray, dict]:
    correction = _find_dominant_wall_angle(img)
    if correction is None or abs(correction) < 0.5:
        return img, meta
    rotated = _rotate_map(img, correction)
    res = float(str(meta.get("resolution", 0.05)).strip())
    origin_raw = meta.get("origin", [0.0, 0.0, 0.0])
    if isinstance(origin_raw, list):
        ox, oy = float(origin_raw[0]), float(origin_raw[1])
    else:
        parts = str(origin_raw).strip("[]").split(",")
        ox, oy = float(parts[0]), float(parts[1])
    dw = (rotated.shape[1] - img.shape[1]) / 2
    dh = (rotated.shape[0] - img.shape[0]) / 2
    updated = dict(meta)
    updated["origin"] = [round(ox - dw * res, 6), round(oy - dh * res, 6), 0.0]
    return rotated, updated

# ---------------------------------------------------------------------------
# Map saving
# ---------------------------------------------------------------------------

def _occupancy_to_image(msg) -> tuple[np.ndarray, dict]:
    arr = msg.grid
    img = np.where(arr == -1,  205,
          np.where(arr == 0,   254,
          np.where(arr >  0,     0, 205))).astype(np.uint8)
    img = np.flipud(img)
    meta = {
        "resolution": msg.resolution,
        "origin": [round(float(msg.origin.position.x), 6),
                   round(float(msg.origin.position.y), 6), 0.0],
    }
    return img, meta

def _save_map(img: np.ndarray, meta: dict, stem: str, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    pgm = out_dir / f"{stem}.pgm"
    Image.fromarray(img).save(pgm)
    paths["pgm"] = pgm
    print(f"[map] Saved {pgm}")

    res    = meta.get("resolution", 0.05)
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    yaml   = out_dir / f"{stem}.yaml"
    yaml.write_text(f"image: {stem}.pgm\nresolution: {res}\norigin: {origin}\n"
                    f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    paths["yaml"] = yaml
    print(f"[map] Saved {yaml}")

    rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)
    rgb[img == 205] = [180, 190, 210]
    rgb[img == 0]   = [40,  40,  40]
    rgb[img == 254] = [240, 240, 240]
    png = out_dir / f"{stem}.png"
    Image.fromarray(rgb).save(png)
    paths["png"] = png
    print(f"[map] Saved {png}")

    return paths

def save_and_straighten(msg) -> str:
    stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\n[map] {msg.width}x{msg.height} cells at {msg.resolution:.4f}m/cell")
    img, meta = _occupancy_to_image(msg)
    raw_paths = _save_map(img, meta, stem, RAW_DIR)
    print("[straighten] Straightening ...")
    straight_img, straight_meta = _straighten(img, meta)
    straight_paths = _save_map(straight_img, straight_meta, stem + "_straight", maps/corrected)
    print(f"[map] Raw       : {raw_paths['pgm']}")
    print(f"[map] Corrected : {straight_paths['pgm']}")
    print(f"Next step:  python map_tool.py --map {straight_paths['pgm']}")
    return straight_paths[".pgm"]  # <-- return the corrected pgm path

def main():
    map_path = Path("maps/raw/latest_map.npy")
    zones_path = Path("maps/raw/run_20260319_122835_zones.json")
    calculate_entropy(map_path, zones_path)

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Main survey
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zones", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=SURVEY_MINUTES)
    args = parser.parse_args()

    robot_ip = os.environ.get("ROBOT_IP")
    if not robot_ip:
        sys.exit("[error] ROBOT_IP environment variable not set")

    if not args.zones.exists():
        sys.exit(f"zones.json not found: {args.zones}")

    payload    = json.loads(args.zones.read_text())
    start_pose = payload.get("start_pose")
    if not start_pose:
        sys.exit("No start_pose in zones.json — set the start marker first")
    sx, sy = start_pose["world"]
    survey_seconds = args.minutes * 60

    # ── Autoconnect Blueprint ─────────────────────────────────────────────
    global unitree_go2
    unitree_go2 = autoconnect(
        unitree_go2_basic,
        voxel_mapper(voxel_size=0.05),
        cost_mapper(),
        replanning_a_star_planner(),
        wavefront_frontier_explorer(),
    ).global_config(
        robot_ip=robot_ip,
        n_workers=6,
        robot_model="unitree_go2",
        robot_width=0.8,
        robot_rotation_diameter=0.8,
    )

    to_svg(unitree_go2, "assets/go2_blueprint.svg")

    coordinator = unitree_go2.build()

    # ── Subscribe to costmap ─────────────────────────────────────────────
    cost_mapper_inst = coordinator.get_instance(CostMapper)
    if cost_mapper_inst is not None:
        def on_costmap(msg) -> None:
            global _latest_map
            _latest_map = msg
            print(f"\r[map] {msg.width}x{msg.height}  "
                  f"free={msg.free_percent:.0f}%  "
                  f"occupied={msg.occupied_percent:.0f}%    ",
                  end="", flush=True)
        cost_mapper_inst.global_costmap.subscribe(on_costmap)
        print("[setup] CostMapper ready")
    else:
        print("[warn] CostMapper not found — map will not be saved")

    # ── Shutdown handler ────────────────────────────────────────────────
    def shutdown(sig, frame) -> None:
        print("\n[shutdown] Saving map ...")
        if _latest_map is not None:
            save_and_straighten(_latest_map)
        else:
            print("[shutdown] No map received yet")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Wait for modules ───────────────────────────────────────────────
    print("[setup] Waiting for modules to stabilise ...")
    time.sleep(5)

    # ── Planner for start/return ───────────────────────────────────────
    planner = coordinator.get_instance(ReplanningAStarPlanner)
    if planner is None:
        sys.exit("[error] ReplanningAStarPlanner not found")

    # ── Move to start pose ─────────────────────────────────────────────
    print(f"\n[survey] Moving to start pose ({sx:.2f}, {sy:.2f}) ...")
    navigate_to(planner, sx, sy, yaw_deg=90.0, label="start pose", timeout=60.0)
    time.sleep(2.0)

    # ── Start exploration ─────────────────────────────────────────────
    explorer = coordinator.get_instance(WavefrontFrontierExplorer)
    
    if explorer is None:
        sys.exit("[error] WavefrontFrontierExplorer not found")

    
    print(f"[survey] Exploring for {args.minutes:.1f} minutes ...")
    survey_end = time.time() + survey_seconds
    while time.time() < survey_end:
        remaining = survey_end - time.time()
        mins, secs = divmod(int(remaining), 60)
        print(f"\r[survey] Time remaining: {mins:02d}:{secs:02d}    ", end="", flush=True)
        time.sleep(1)

    explorer.stop_exploration()

    # ── Return to start pose ───────────────────────────────────────────
    print(f"\n[survey] Returning to start pose ({sx:.2f}, {sy:.2f}) ...")
    navigate_to(planner, sx, sy, yaw_deg=90.0, label="return start pose", timeout=90.0)

    # ── Save map ──────────────────────────────────────────────────────
    print("\n[survey] Saving map ...")
    if _latest_map is not None:
        pgm_path = save_and_straighten(_latest_map)
        zones_path = Path("maps/raw/run_20260319_122835_zones.json")

        from entropy import calculate_entropy

        calculate_entropy(pgm_path, zones_path)
    else:
        print("[warn] No map received")

    print("\n[survey] Done.")

if __name__ == "__main__":
    main()
