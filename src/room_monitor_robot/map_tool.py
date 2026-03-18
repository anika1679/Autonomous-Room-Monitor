"""
map_tool.py  —  Interactive map tool for the dimos / Go2 entropy pipeline.
 
TWO MODES:
 
  1. AFTER A RUN  (--map path/to/map.pgm)
     Load a saved map, draw 6 zone rectangles, name them, save zones.json.
     This is the normal workflow — robot runs first, then you zone the map.
 
  2. LIVE (--live)
     Subscribe to ROS2 /map topic, show the map as it builds, let you
     click a start point (publishes /initialpose), and save the map on exit.
 
USAGE:
  # After a run — zone an existing map:
  python map_tool.py --map ./maps/run_20250318_143022.pgm
 
  # Live with ROS2 running:
  source /ros2_ws/install/setup.bash
  python map_tool.py --live
 
MOUSE CONTROLS:
  Left-click           : (live mode) set robot start pose
  Right-click + drag   : draw a zone rectangle
  Double-click a zone  : rename it
  Delete / Backspace   : remove last zone
 
KEYBOARD:
  S   : save map + zones now
  C   : clear all zones
  Q   : save and quit
 
OUTPUT  (written to ./maps/):
  run_<timestamp>.pgm           ROS-compatible map image
  run_<timestamp>.yaml          ROS map metadata
  run_<timestamp>.png           colour map with zone overlays
  run_<timestamp>_zones.json    zone definitions (pixel + world coords)
                                → fed directly into entropy_skill.py
"""
 
from __future__ import annotations
 
import argparse
import json
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import simpledialog, messagebox
from typing import Optional
 
import numpy as np
 
try:
    from PIL import Image, ImageDraw, ImageTk, ImageFont
except ImportError:
    sys.exit("Install Pillow:  pip install Pillow")
 
 
MAPS_DIR     = Path("./maps")
MAX_ZONES    = 6
DISPLAY_MAX  = 800   # max pixels for display (will scale map to fit)
 
# Zone colours (RGBA) — one per zone slot
ZONE_COLORS = [
    (255, 80,  80,  120),   # red
    (80,  180, 255, 120),   # blue
    (80,  220, 120, 120),   # green
    (255, 200, 60,  120),   # amber
    (200, 100, 255, 120),   # purple
    (255, 140, 60,  120),   # orange
]
ZONE_BORDER = [
    (200, 40,  40),
    (40,  120, 200),
    (40,  160, 80),
    (200, 150, 20),
    (150, 60,  200),
    (200, 90,  20),
]
 
 
# ---------------------------------------------------------------------------
# Map I/O helpers
# ---------------------------------------------------------------------------
 
def load_pgm(path: Path) -> tuple[np.ndarray, dict]:
    """Load a ROS .pgm map and its matching .yaml metadata."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    yaml_path = path.with_suffix(".yaml")
    meta: dict = {}
    if yaml_path.exists():
        import yaml   # pip install pyyaml
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
    return arr, meta
 
 
def save_map(arr: np.ndarray, meta: dict, stem: str) -> dict[str, Path]:
    """Save map array as .pgm, .yaml, and annotated .png. Returns paths."""
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
 
    # .pgm
    pgm_path = MAPS_DIR / f"{stem}.pgm"
    Image.fromarray(arr.astype(np.uint8)).save(pgm_path)
    paths["pgm"] = pgm_path
 
    # .yaml
    yaml_path = MAPS_DIR / f"{stem}.yaml"
    resolution = meta.get("resolution", 0.05)
    origin     = meta.get("origin", [0.0, 0.0, 0.0])
    yaml_text  = (
        f"image: {stem}.pgm\n"
        f"resolution: {resolution}\n"
        f"origin: {origin}\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_text)
    paths["yaml"] = yaml_path
 
    return paths
 
 
def save_annotated_png(
    arr: np.ndarray,
    zones: list[dict],
    stem: str,
    start_px: Optional[tuple[int, int]] = None,
) -> Path:
    """Save a colour PNG with zone overlays and optional start marker."""
    rgb = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
 
    for i, zone in enumerate(zones):
        color  = ZONE_COLORS[i % len(ZONE_COLORS)]
        border = ZONE_BORDER[i % len(ZONE_BORDER)]
        x0, y0, x1, y1 = zone["pixel_rect"]
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=border + (255,), width=2)
        # Zone label
        draw.text((x0 + 4, y0 + 4), zone["name"], fill=(0, 0, 0, 220))
 
    if start_px:
        sx, sy = start_px
        r = 6
        draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                     fill=(0, 220, 80, 230), outline=(0, 0, 0, 255), width=2)
        draw.text((sx + r + 2, sy - 6), "start", fill=(0, 0, 0, 220))
 
    composed = Image.alpha_composite(rgb, overlay).convert("RGB")
    png_path = MAPS_DIR / f"{stem}.png"
    composed.save(png_path)
    return png_path
 
 
def save_zones_json(
    zones: list[dict],
    meta: dict,
    stem: str,
    start_px: Optional[tuple[int, int]] = None,
    map_shape: Optional[tuple[int, int]] = None,
) -> Path:
    """
    Save zones.json — this is what entropy_skill.py reads.
    Each zone includes pixel_rect and world_rect (metres, map frame).
    """
    resolution = meta.get("resolution", 0.05)
    origin     = meta.get("origin", [0.0, 0.0, 0.0])
    h          = map_shape[0] if map_shape else 0
 
    def px_to_world(px: int, py: int) -> tuple[float, float]:
        wx = origin[0] + px * resolution
        wy = origin[1] + (h - py) * resolution
        return round(wx, 4), round(wy, 4)
 
    out_zones = []
    for i, zone in enumerate(zones):
        x0, y0, x1, y1 = zone["pixel_rect"]
        wx0, wy0 = px_to_world(x0, y0)
        wx1, wy1 = px_to_world(x1, y1)
        cx_px = (x0 + x1) // 2
        cy_px = (y0 + y1) // 2
        cx_w, cy_w = px_to_world(cx_px, cy_px)
        out_zones.append({
            "zone_id":    i + 1,
            "name":       zone["name"],
            "pixel_rect": [x0, y0, x1, y1],
            "world_rect": [wx0, wy0, wx1, wy1],
            "centroid_world": [cx_w, cy_w],
        })
 
    payload: dict = {
        "map_stem":   stem,
        "resolution": resolution,
        "origin":     origin,
        "zones":      out_zones,
    }
    if start_px and map_shape:
        swx, swy = px_to_world(start_px[0], start_px[1])
        payload["start_pose"] = {
            "pixel":       list(start_px),
            "world":       [swx, swy],
        }
 
    json_path = MAPS_DIR / f"{stem}_zones.json"
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path
 
 
# ---------------------------------------------------------------------------
# Tkinter map UI
# ---------------------------------------------------------------------------
 
class MapUI:
    def __init__(
        self,
        root: tk.Tk,
        map_arr: np.ndarray,
        meta: dict,
        live: bool = False,
    ) -> None:
        self.root      = root
        self.map_arr   = map_arr
        self.meta      = meta
        self.live      = live
        self.zones: list[dict] = []
        self.start_px: Optional[tuple[int, int]] = None
 
        # Drawing state
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_rect  = None
 
        # Compute display scale
        h, w = map_arr.shape[:2]
        scale = min(DISPLAY_MAX / w, DISPLAY_MAX / h, 1.0)
        self.scale  = scale
        self.dw     = int(w * scale)
        self.dh     = int(h * scale)
 
        self._build_ui()
        self._render()
 
    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
 
    def _build_ui(self) -> None:
        self.root.title("dimos map tool — zone editor")
        self.root.resizable(False, False)
 
        # Top bar
        bar = tk.Frame(self.root, bg="#1e1e2e", pady=6)
        bar.pack(fill=tk.X)
 
        mode_label = "LIVE" if self.live else "MAP EDITOR"
        tk.Label(bar, text=f"  dimos map tool  [{mode_label}]",
                 bg="#1e1e2e", fg="#cdd6f4", font=("monospace", 11, "bold")).pack(side=tk.LEFT)
 
        tk.Button(bar, text="Save  [S]", command=self.save_all,
                  bg="#313244", fg="#a6e3a1", relief=tk.FLAT,
                  padx=10).pack(side=tk.RIGHT, padx=4)
        tk.Button(bar, text="Clear zones  [C]", command=self.clear_zones,
                  bg="#313244", fg="#fab387", relief=tk.FLAT,
                  padx=10).pack(side=tk.RIGHT, padx=4)
 
        # Canvas
        self.canvas = tk.Canvas(
            self.root, width=self.dw, height=self.dh,
            bg="#11111b", highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack()
 
        # Status bar
        self.status_var = tk.StringVar(value="Right-click + drag to draw zones.  Q = save & quit.")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#181825", fg="#a6adc8", font=("monospace", 9),
                 anchor="w", padx=8).pack(fill=tk.X)
 
        # Zone list sidebar
        sidebar = tk.Frame(self.root, bg="#1e1e2e")
        sidebar.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(sidebar, text="Zones (double-click to rename):",
                 bg="#1e1e2e", fg="#cdd6f4", font=("monospace", 9)).pack(anchor="w")
        self.zone_listbox = tk.Listbox(
            sidebar, height=MAX_ZONES, bg="#181825", fg="#cdd6f4",
            font=("monospace", 9), selectbackground="#313244",
            relief=tk.FLAT, highlightthickness=0,
        )
        self.zone_listbox.pack(fill=tk.X)
        self.zone_listbox.bind("<Double-Button-1>", self._rename_zone)
 
        # Bindings
        self.canvas.bind("<Button-1>",        self._on_left_click)
        self.canvas.bind("<Button-3>",        self._on_right_press)
        self.canvas.bind("<B3-Motion>",       self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.root.bind("<s>", lambda _: self.save_all())
        self.root.bind("<S>", lambda _: self.save_all())
        self.root.bind("<c>", lambda _: self.clear_zones())
        self.root.bind("<C>", lambda _: self.clear_zones())
        self.root.bind("<q>", lambda _: self._quit())
        self.root.bind("<Q>", lambda _: self._quit())
        self.root.bind("<Delete>",    lambda _: self._remove_last_zone())
        self.root.bind("<BackSpace>", lambda _: self._remove_last_zone())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
 
    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
 
    def _render(self) -> None:
        """Re-render the map with current zones and start marker."""
        arr = self.map_arr
        h, w = arr.shape[:2]
 
        # Convert occupancy grid (0=free=white, 100=occ=black, -1=unknown=grey)
        # ROS convention: free=254, occ=0, unknown=205 in saved .pgm
        display = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
        display = display.resize((self.dw, self.dh), Image.NEAREST)
 
        overlay = Image.new("RGBA", (self.dw, self.dh), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
 
        for i, zone in enumerate(self.zones):
            color  = ZONE_COLORS[i % len(ZONE_COLORS)]
            border = ZONE_BORDER[i % len(ZONE_BORDER)]
            x0, y0, x1, y1 = self._world_rect_to_display(zone["pixel_rect"])
            draw.rectangle([x0, y0, x1, y1],
                           fill=color, outline=border + (255,), width=2)
            draw.text((x0 + 4, y0 + 3), zone["name"], fill=(0, 0, 0, 230))
 
        if self.start_px:
            sx = int(self.start_px[0] * self.scale)
            sy = int(self.start_px[1] * self.scale)
            r = 7
            draw.ellipse([sx-r, sy-r, sx+r, sy+r],
                         fill=(0, 220, 80, 220), outline=(0,0,0,255), width=2)
 
        composed = Image.alpha_composite(display, overlay)
        self._tk_img = ImageTk.PhotoImage(composed)
        self.canvas.delete("map")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img, tags="map")
        self.canvas.tag_lower("map")
 
        # Redraw drag rect on top if active
        if self._drag_rect:
            self.canvas.tag_raise(self._drag_rect)
 
        self._update_zone_list()
 
    def _world_rect_to_display(self, pixel_rect: list[int]) -> tuple:
        x0, y0, x1, y1 = pixel_rect
        return (int(x0 * self.scale), int(y0 * self.scale),
                int(x1 * self.scale), int(y1 * self.scale))
 
    def _display_to_pixel(self, dx: int, dy: int) -> tuple[int, int]:
        return int(dx / self.scale), int(dy / self.scale)
 
    def _update_zone_list(self) -> None:
        self.zone_listbox.delete(0, tk.END)
        for i, zone in enumerate(self.zones):
            x0, y0, x1, y1 = zone["pixel_rect"]
            self.zone_listbox.insert(
                tk.END,
                f"  Zone {i+1}: {zone['name']:<16}  "
                f"[{x0},{y0} → {x1},{y1}]"
            )
 
    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
 
    def _on_left_click(self, event: tk.Event) -> None:
        """Left-click: set start pose (live mode) or just mark it."""
        px, py = self._display_to_pixel(event.x, event.y)
        self.start_px = (px, py)
        if self.live:
            self._publish_initial_pose(px, py)
            self._set_status(f"Start pose set at pixel ({px}, {py}) — published to /initialpose")
        else:
            self._set_status(f"Start marker placed at pixel ({px}, {py})")
        self._render()
 
    def _on_right_press(self, event: tk.Event) -> None:
        self._drag_start = (event.x, event.y)
        self._drag_rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#f38ba8", width=2, dash=(4, 2),
        )
 
    def _on_right_drag(self, event: tk.Event) -> None:
        if self._drag_start and self._drag_rect:
            x0, y0 = self._drag_start
            self.canvas.coords(self._drag_rect, x0, y0, event.x, event.y)
 
    def _on_right_release(self, event: tk.Event) -> None:
        if not self._drag_start:
            return
        if len(self.zones) >= MAX_ZONES:
            messagebox.showwarning("Max zones", f"Maximum {MAX_ZONES} zones allowed.")
            self.canvas.delete(self._drag_rect)
            self._drag_rect = None
            self._drag_start = None
            return
 
        x0d, y0d = self._drag_start
        x1d, y1d = event.x, event.y
        # Normalise so x0 < x1
        x0d, x1d = min(x0d, x1d), max(x0d, x1d)
        y0d, y1d = min(y0d, y1d), max(y0d, y1d)
 
        if abs(x1d - x0d) < 10 or abs(y1d - y0d) < 10:
            self.canvas.delete(self._drag_rect)
            self._drag_rect = None
            self._drag_start = None
            return
 
        # Convert to pixel coords
        x0, y0 = self._display_to_pixel(x0d, y0d)
        x1, y1 = self._display_to_pixel(x1d, y1d)
 
        zone_num = len(self.zones) + 1
        default_name = f"Zone {zone_num}"
        name = simpledialog.askstring(
            "Name this zone",
            f"Zone {zone_num} name:",
            initialvalue=default_name,
            parent=self.root,
        ) or default_name
 
        self.zones.append({"name": name, "pixel_rect": [x0, y0, x1, y1]})
        self._set_status(
            f"Zone '{name}' added ({len(self.zones)}/{MAX_ZONES}).  "
            f"{'Draw more zones.' if len(self.zones) < MAX_ZONES else 'All zones defined — press S to save.'}"
        )
 
        self.canvas.delete(self._drag_rect)
        self._drag_rect = None
        self._drag_start = None
        self._render()
 
    def _rename_zone(self, event: tk.Event) -> None:
        sel = self.zone_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        new_name = simpledialog.askstring(
            "Rename zone",
            f"New name for zone {idx+1}:",
            initialvalue=self.zones[idx]["name"],
            parent=self.root,
        )
        if new_name:
            self.zones[idx]["name"] = new_name
            self._render()
 
    # ------------------------------------------------------------------
    # Zone management
    # ------------------------------------------------------------------
 
    def clear_zones(self) -> None:
        self.zones.clear()
        self._set_status("All zones cleared.")
        self._render()
 
    def _remove_last_zone(self) -> None:
        if self.zones:
            removed = self.zones.pop()
            self._set_status(f"Removed zone '{removed['name']}'.")
            self._render()
 
    # ------------------------------------------------------------------
    # Live mode: publish initial pose to ROS2
    # ------------------------------------------------------------------
 
    def _publish_initial_pose(self, px: int, py: int) -> None:
        """Publish a PoseWithCovarianceStamped to /initialpose (AMCL)."""
        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PoseWithCovarianceStamped
 
            resolution = self.meta.get("resolution", 0.05)
            origin     = self.meta.get("origin", [0.0, 0.0, 0.0])
            h          = self.map_arr.shape[0]
            wx = origin[0] + px * resolution
            wy = origin[1] + (h - py) * resolution
 
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node("map_tool_pose_publisher")
            pub  = node.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
 
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.pose.pose.position.x = wx
            msg.pose.pose.position.y = wy
            msg.pose.pose.orientation.w = 1.0
            msg.pose.covariance[0]  = 0.25
            msg.pose.covariance[7]  = 0.25
            msg.pose.covariance[35] = 0.06
 
            # Publish a few times to make sure it's received
            for _ in range(3):
                pub.publish(msg)
                time.sleep(0.05)
 
            node.destroy_node()
        except Exception as e:
            self._set_status(f"[warn] Could not publish initialpose: {e}")
 
    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
 
    def save_all(self) -> None:
        stem = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        map_paths = save_map(self.map_arr, self.meta, stem)
        png_path  = save_annotated_png(
            self.map_arr, self.zones, stem, self.start_px
        )
        json_path = save_zones_json(
            self.zones, self.meta, stem, self.start_px,
            map_shape=self.map_arr.shape[:2],
        )
        msg = (
            f"Saved to ./maps/:\n"
            f"  {stem}.pgm\n"
            f"  {stem}.yaml\n"
            f"  {stem}.png\n"
            f"  {stem}_zones.json\n\n"
            f"zones.json is ready for entropy_skill.py"
        )
        self._set_status(f"Saved: {stem}.pgm/yaml/png + _zones.json")
        messagebox.showinfo("Saved", msg)
 
    def _quit(self) -> None:
        self.save_all()
        self.root.destroy()
 
    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()
 
    # ------------------------------------------------------------------
    # Live map updates (called from ROS thread)
    # ------------------------------------------------------------------
 
    def update_map(self, arr: np.ndarray, meta: dict) -> None:
        self.map_arr = arr
        self.meta    = meta
        self.root.after(0, self._render)
 
 
# ---------------------------------------------------------------------------
# Live ROS2 map subscriber (runs in background thread)
# ---------------------------------------------------------------------------
 
def start_live_map_subscriber(ui: MapUI) -> None:
    """Subscribe to /map and push updates to the UI."""
    try:
        import rclpy
        from rclpy.node import Node
        from nav_msgs.msg import OccupancyGrid
    except ImportError:
        print("[warn] rclpy not available — live mode disabled")
        return
 
    def ros_thread() -> None:
        rclpy.init()
        node = rclpy.create_node("map_tool_subscriber")
 
        def map_callback(msg: OccupancyGrid) -> None:
            w, h = msg.info.width, msg.info.height
            # OccupancyGrid: -1=unknown, 0=free, 100=occupied
            # Convert to image: free=254, occ=0, unknown=205
            data  = np.array(msg.data, dtype=np.int8).reshape(h, w)
            image = np.where(data == -1, 205,
                    np.where(data == 0,  254,
                    np.where(data == 100, 0, 205))).astype(np.uint8)
            # ROS maps are stored bottom-up
            image = np.flipud(image)
            meta = {
                "resolution": msg.info.resolution,
                "origin": [
                    msg.info.origin.position.x,
                    msg.info.origin.position.y,
                    0.0,
                ],
            }
            ui.update_map(image, meta)
 
        node.create_subscription(OccupancyGrid, "/map", map_callback, 10)
        rclpy.spin(node)
 
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()
 
 
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(description="dimos map zone editor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--map",  type=Path, metavar="FILE",
                       help="Load an existing .pgm map file")
    group.add_argument("--live", action="store_true",
                       help="Subscribe to live /map topic (requires ROS2)")
    args = parser.parse_args()
 
    if args.map:
        if not args.map.exists():
            sys.exit(f"Map file not found: {args.map}")
        arr, meta = load_pgm(args.map)
        print(f"Loaded map: {args.map}  shape={arr.shape}")
    else:
        # Live mode — start with blank map, ROS will fill it in
        arr  = np.full((400, 400), 205, dtype=np.uint8)   # all-unknown grey
        meta = {"resolution": 0.05, "origin": [0.0, 0.0, 0.0]}
        print("Live mode — waiting for /map topic ...")
 
    root = tk.Tk()
    ui   = MapUI(root, arr, meta, live=args.live)
 
    if args.live:
        start_live_map_subscriber(ui)
 
    root.mainloop()
 
 
if __name__ == "__main__":
    main()
 
