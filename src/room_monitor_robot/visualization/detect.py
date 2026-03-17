"""
Live webcam object detection using YOLO.
"""
import argparse
import sys

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Object detection from webcam")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model (e.g. yolov8n.pt)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    return p.parse_args()


def run():
    args = parse_args()
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.", file=sys.stderr)
        sys.exit(1)

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=args.conf, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )

        cv2.imshow("Object detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
