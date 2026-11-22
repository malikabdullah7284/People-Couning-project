"""
People Counter v3 - stronger detection + less double-counting
---------------------------------------------------------------
Fixes over v2:
  1. Stronger model: yolov8s.pt (change to yolov8m.pt for even stronger,
     slower) and imgsz=1280, so small / far people in a video are detected.
  2. BoT-SORT tracker WITH Re-ID (appearance matching): it remembers a
     person's appearance pattern, so after a short hide / overlap it
     tries to give them back the SAME id instead of a new one.
  3. Longer track_buffer (90 frames): a person who disappears for up to
     ~3 seconds and comes back keeps their old id.
  4. Confirmation rule: an id is counted in the TOTAL only after it has
     been seen in at least 5 frames. One-frame flicker / false detections
     are ignored, so they don't inflate your total.

Honest limit: if a person leaves the video completely and returns after
a long time, or is hidden for a long time, any tracker can still give
them a new id. Counting a person exactly once forever, even after long
exits, needs a face/appearance database - that is a next-level feature.

Usage:
  python people_counter.py --source 0                 (webcam)
  python people_counter.py --source "test.mp4"        (video file)
  python people_counter.py --source "test.mp4" --save (also save output)
Press Q to quit early.
"""

import argparse
import os

import cv2
from ultralytics import YOLO

# --- Settings you can change -------------------------------------------
MODEL_NAME = "yolov8s.pt"   # stronger than yolov8n. Even stronger: "yolov8m.pt"
CONFIDENCE = 0.30           # lower = catches more people, a few more mistakes
IMAGE_SIZE = 1280           # bigger = better for small/far people, slower
MIN_FRAMES_TO_COUNT = 5     # id must be seen this many frames to be counted
TRACK_BUFFER = 90           # frames to remember a lost person (~3 sec at 30fps)
# ------------------------------------------------------------------------

TRACKER_FILE = "my_botsort.yaml"
TRACKER_YAML = f"""tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: {TRACK_BUFFER}
match_thresh: 0.80
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.50
appearance_thresh: 0.25
with_reid: True
model: auto
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help='0 for webcam, or a video file path like "test.mp4"')
    parser.add_argument("--save", action="store_true",
                        help="save the output video as output_counted.mp4")
    args = parser.parse_args()

    source = args.source
    if source is None:
        print("1 = Webcam   2 = Video file")
        choice = input("Type 1 or 2 and press Enter: ").strip()
        source = input("Paste the video file path: ").strip().strip('"') if choice == "2" else "0"

    # Write our stronger tracker config once, then use it
    if not os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "w") as f:
            f.write(TRACKER_YAML)

    cap = cv2.VideoCapture(0 if str(source) == "0" else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}. Check the file path / camera permission.")

    model = YOLO(MODEL_NAME)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("output_counted.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames_seen = {}   # track_id -> how many frames we have seen this id
    confirmed = set()  # ids that passed MIN_FRAMES_TO_COUNT = real people
    max_in_frame = 0

    print("People counter v3 running... press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.track(
            frame,
            persist=True,
            tracker=TRACKER_FILE,
            classes=[0],               # 0 = person only
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )[0]

        boxes = result.boxes
        in_frame_now = 0

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            in_frame_now = len(boxes)
            ids = boxes.id.tolist()
            coords = boxes.xyxy.tolist()

            for track_id, box in zip(ids, coords):
                track_id = int(track_id)
                frames_seen[track_id] = frames_seen.get(track_id, 0) + 1

                # Count this person in the total ONLY once,
                # and only after we are sure they are real
                if frames_seen[track_id] >= MIN_FRAMES_TO_COUNT:
                    confirmed.add(track_id)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        max_in_frame = max(max_in_frame, in_frame_now)

        cv2.rectangle(frame, (10, 10), (470, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"People in frame now: {in_frame_now}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total unique people: {len(confirmed)}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("People Counter v3 - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n========== RESULT ==========")
    print(f"Total unique people (counted once each): {len(confirmed)}")
    print(f"Most people in one frame               : {max_in_frame}")
    print("============================")
    if args.save:
        print("Saved counted video as: output_counted.mp4")


if __name__ == "__main__":
    main()
