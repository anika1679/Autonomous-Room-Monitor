# entropy.py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import csv

def pgm_to_array(pgm_path):
    """Convert a PGM image to occupancy array with values: 0=free, 1=occupied, -1=unknown"""
    img = np.array(Image.open(pgm_path))
    arr = np.full(img.shape, -1, dtype=int)
    arr[img == 0] = 0      # free
    arr[img == 255] = 1    # occupied
    return arr

def shannon_entropy(arr):
    arr = arr[arr != -1]  # remove unknowns
    if arr.size == 0:
        return 0.0
    p_occ = np.mean(arr == 1)
    p_free = np.mean(arr == 0)
    probs = [p for p in [p_occ, p_free] if p > 0]
    return -sum(p * np.log2(p) for p in probs)

def calculate_zone_entropy(map_array, zones):
    zone_entropies = []
    zone_sizes = []
    for zone in zones:
        x_min, y_min, x_max, y_max = zone["pixel_rect"]
        subarray = map_array[y_min:y_max, x_min:x_max]
        entropy = shannon_entropy(subarray)
        zone_entropies.append(entropy)
        zone_sizes.append(subarray.size)
    return np.array(zone_entropies), np.array(zone_sizes)

def calculate_entropy(pgm_path, zones_path, csv_file="entropy_over_time.csv"):
    pgm_path = Path(pgm_path)
    zones_path = Path(zones_path)

    if not pgm_path.exists():
        raise FileNotFoundError(f"Map file not found: {pgm_path}")
    if not zones_path.exists():
        raise FileNotFoundError(f"Zones file not found: {zones_path}")

    map_array = pgm_to_array(pgm_path)
    zones = json.loads(zones_path.read_text())["zones"]

    zone_entropies, zone_sizes = calculate_zone_entropy(map_array, zones)
    overall_entropy = np.mean(zone_entropies)
    weighted_entropy = np.sum(zone_entropies * zone_sizes / np.sum(zone_sizes))

    timestamp = datetime.now().isoformat()
    row = [timestamp, overall_entropy, weighted_entropy] + list(zone_entropies)

    if not Path(csv_file).exists():
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["timestamp", "overall_entropy", "weighted_entropy"] + [f"zone_{i}" for i in range(len(zone_entropies))]
            writer.writerow(header)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(f"[entropy] Saved entropies at {timestamp}")
    return zone_entropies, overall_entropy, weighted_entropy
