import cv2
import numpy as np
import time
import threading
import argparse
import os
import sys

# ─── SOUND ───────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("[WARN] pygame not installed — no alarm sound. Run: pip install pygame")

# ─── PHONE NOTIFICATION (ntfy.sh) ──────────────────
try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[WARN] requests not installed. Run: pip install requests")

# ─── DETECTION MODEL ──────────────────────────────────────────
USE_YOLO8 = False
USE_YOLO3 = False
net = None
yolo8_model = None
yolo3_classes = []

def load_model():
    global USE_YOLO8, USE_YOLO3, net, yolo8_model, yolo3_classes

    # 1) Try YOLOv8
    yolo8_path = "yolov8n.pt"
    try:
        from ultralytics import YOLO
        yolo8_model = YOLO(yolo8_path)
        USE_YOLO8 = True
        print(f"[OK] YOLOv8 loaded: {yolo8_path}")
        return
    except Exception as e:
        print(f"[INFO] YOLOv8 failed ({e}), trying YOLOv3-tiny...")

    # 2) Try YOLOv3-tiny
    cfg_paths     = ["yolov3-tiny.cfg", "yolov3-tiny/yolov3-tiny.cfg"]
    weight_paths  = ["yolov3-tiny.weights", "yolov3-tiny/yolov3-tiny.weights"]
    coco_paths    = ["coco.names", "yolov3-tiny/coco.names"]

    for cfg, wt, cn in zip(cfg_paths, weight_paths, coco_paths):
        if os.path.exists(cfg) and os.path.exists(wt):
            try:
                net = cv2.dnn.readNet(wt, cfg)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                if os.path.exists(cn):
                    with open(cn) as f:
                        yolo3_classes = [l.strip() for l in f.readlines()]
                else:
                    yolo3_classes = ["person"] * 80
                USE_YOLO3 = True
                print(f"[OK] YOLOv3-tiny loaded: {cfg}")
                return
            except Exception as e:
                print(f"[WARN] YOLOv3-tiny load error: {e}")

    # 3) HOG fallback
    print("[WARN] No YOLO model found — using HOG detector (less accurate).")
    print("       Put yolov8n.pt or yolov3-tiny.cfg/.weights in the same folder.")

load_model()


# ─── DETECT PERSONS ────────────────────────────────────────────────────────────
def detect_persons(frame):
    """Returns list of [x1, y1, x2, y2] for each detected person."""
    boxes = []

    if USE_YOLO8:
        results = yolo8_model(frame, classes=[0], conf=0.40, verbose=False)
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # Ensure valid box
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])

    elif USE_YOLO3:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)

        raw_boxes, confidences = [], []
        for out in outputs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # class 0 = person in COCO
                if class_id == 0 and confidence > 0.40:
                    cx = int(det[0] * w)
                    cy = int(det[1] * h)
                    bw = int(det[2] * w)
                    bh = int(det[3] * h)
                    x1 = max(0, cx - bw // 2)
                    y1 = max(0, cy - bh // 2)
                    x2 = min(w, cx + bw // 2)
                    y2 = min(h, cy + bh // 2)
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(float(confidence))

        # NMS to remove duplicate boxes
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, 0.40, 0.40)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = raw_boxes[i]
                boxes.append([x, y, x + bw, y + bh])

    else:
        # HOG fallback
        resized = cv2.resize(frame, (640, 480))
        rects, _ = cv2.HOGDescriptor().detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
        sx = frame.shape[1] / 640
        sy = frame.shape[0] / 480
        for (x, y, bw, bh) in rects:
            boxes.append([int(x*sx), int(y*sy),
                          int((x+bw)*sx), int((y+bh)*sy)])

    return boxes

# ─── IOU TRACKER ────────────────────────────────────────
class Track:
    def __init__(self, track_id, bbox):
        self.id       = track_id
        self.bbox     = bbox                        # [x1,y1,x2,y2]
        self.centroid = self._centroid(bbox)
        self.side     = None                        # 'left' or 'right'
        self.missed   = 0
        self.active   = True

    @staticmethod
    def _centroid(bbox):
        return ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)

    def update(self, bbox):
        self.bbox     = bbox
        self.centroid = self._centroid(bbox)
        self.missed   = 0
        self.active   = True


def iou(b1, b2):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


class Tracker:
    def __init__(self, iou_thresh=0.25, max_missed=30):
        self.tracks     = {}          # id -> Track
        self.next_id    = 0
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2]
        returns: list of Track objects (active)
        """
        if not detections:
            for t in self.tracks.values():
                t.missed += 1
            self._cleanup()
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())

        if not track_ids:
            for det in detections:
                self.tracks[self.next_id] = Track(self.next_id, det)
                self.next_id += 1
            return list(self.tracks.values())

        # Build IOU matrix  [tracks × detections]
        iou_mat = np.zeros((len(track_ids), len(detections)))
        for ti, tid in enumerate(track_ids):
            for di, det in enumerate(detections):
                iou_mat[ti, di] = iou(self.tracks[tid].bbox, det)

        # Greedy match highest IOU first
        matched_t = set()
        matched_d = set()

        # Sort by descending IOU
        flat = sorted(
            [(iou_mat[ti, di], ti, di)
             for ti in range(len(track_ids))
             for di in range(len(detections))],
            reverse=True
        )
        for score, ti, di in flat:
            if ti in matched_t or di in matched_d:
                continue
            if score < self.iou_thresh:
                break
            self.tracks[track_ids[ti]].update(detections[di])
            matched_t.add(ti)
            matched_d.add(di)

        # Unmatched tracks → increment missed
        for ti, tid in enumerate(track_ids):
            if ti not in matched_t:
                self.tracks[tid].missed += 1

        # Unmatched detections → new tracks
        for di, det in enumerate(detections):
            if di not in matched_d:
                self.tracks[self.next_id] = Track(self.next_id, det)
                self.next_id += 1

        self._cleanup()
        return list(self.tracks.values())

    def _cleanup(self):
        dead = [tid for tid, t in self.tracks.items()
                if t.missed > self.max_missed]
        for tid in dead:
            del self.tracks[tid]


# ─── ALARM SOUND (siren using pygame + numpy) ─────────────────────────────────
_alarm_running = False
_alarm_thread  = None

def _make_siren_sound(sample_rate=44100, duration=0.8):
    """Generate a sharp police-style wee-woo siren."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Alternate sharply between 800 Hz and 1600 Hz (police wee-woo)
    freq = np.where((t % 0.4) < 0.2, 800.0, 1600.0)
    phase = np.cumsum(2 * np.pi * freq / sample_rate)
    wave  = np.sin(phase)
    # Add slight harmonic for richness
    wave += 0.3 * np.sin(2 * phase)
    wave  = wave / np.max(np.abs(wave))   # normalize
    wave  = (wave * 30000).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([wave, wave]))
    return stereo

def _alarm_loop():
    global _alarm_running
    if not HAS_SOUND:
        while _alarm_running:
            print("\a", end="", flush=True)   # terminal beep fallback
            time.sleep(0.5)
        return
    siren = _make_siren_sound()
    sound = pygame.sndarray.make_sound(siren)
    while _alarm_running:
        sound.play()
        time.sleep(0.65)

def start_alarm():
    global _alarm_running, _alarm_thread
    if _alarm_running:
        return
    _alarm_running = True
    _alarm_thread = threading.Thread(target=_alarm_loop, daemon=True)
    _alarm_thread.start()

def stop_alarm():
    global _alarm_running
    _alarm_running = False
    if HAS_SOUND:
        pygame.mixer.stop()


# ─── PHONE NOTIFICATION via ntfy.sh (free) ────────────────────────────────────
def send_phone_notification(topic, message="EMERGENCY ALERT! People Counting System"):
    if not topic:
        print("[NOTIFY] ✗ No topic set. Run with:  python people_counter.py --notify TOPIC_NAME")
        return
    if not HAS_REQUESTS:
        print("[NOTIFY] ✗ 'requests' not installed. Run: pip install requests")
        return

    clean_topic = topic.strip().lower().replace(" ", "")
    url = f"https://ntfy.sh/{clean_topic}"
    print(f"[NOTIFY] Sending → {url}  ...")
    try:
        resp = _req.post(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title":    "EMERGENCY ALERT",
                "Priority": "urgent",
                "Tags":     "rotating_light,sos",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"[NOTIFY] ✓ SUCCESS! Phone notification sent to topic: '{clean_topic}'")
        else:
            print(f"[NOTIFY] ✗ Failed — HTTP {resp.status_code}")
            print(f"[NOTIFY]   Response: {resp.text[:200]}")
            print("[NOTIFY]   Tip: topic name sirf lowercase letters aur numbers hone chahiye")
    except Exception as e:
        print(f"[NOTIFY] ✗ Error: {e}")
        print("[NOTIFY]   Internet connection check karo aur dobara try karo.")


def send_notification_async(topic, message):
    """Background thread mein notification bhejo — video freeze na ho."""
    t = threading.Thread(
        target=send_phone_notification,
        args=(topic, message),
        daemon=True
    )
    t.start()


def test_notification(topic):
    """Startup pe test notification bhejo taaki pta chale notification kaam kar rahi hai."""
    if not topic:
        return
    print("\n" + "="*55)
    print("  NOTIFICATION TEST — program shuru hote hi bhej raha hai")
    print("="*55)
    send_notification_async(topic, f"✅ PEOPLE DETECTION STARTED! Topic: {topic}")
    print("="*55 + "\n")


# ─── DRAWING HELPERS ──────────────────────────────────────────────────────────
FONT = cv2.FONT_HERSHEY_DUPLEX

def draw_label(frame, text, pos, font_scale=0.6, text_color=(255,255,255),
               bg_color=(0,0,0), thickness=1, pad=5):
    """Text with filled background rectangle."""
    (tw, th), bl = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale,
                text_color, thickness, cv2.LINE_AA)
    return tw + 2 * pad, th + 2 * pad


def draw_counting_line(frame, line_x, video_h):
    """Vertical counting line — just the line + tiny arrows only."""
    # Cyan vertical line
    cv2.line(frame, (line_x, 0), (line_x, video_h), (0, 220, 220), 2)

    # Tiny arrows directly on the line at mid-height — no big labels
    mid = video_h // 2
    # Small "<" left side
    cv2.putText(frame, "<", (line_x - 18, mid + 6),
                FONT, 0.45, (100, 180, 255), 1, cv2.LINE_AA)
    # Small ">" right side
    cv2.putText(frame, ">", (line_x + 5, mid + 6),
                FONT, 0.45, (100, 255, 120), 1, cv2.LINE_AA)


def draw_alarm_overlay(frame, h, w):
    """Red flashing border when alarm is active."""
    t = time.time()
    alpha = 0.5 + 0.5 * abs(np.sin(t * 5))
    color = (0, 0, int(255 * alpha))
    thickness = 8
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)

    msg = "!! EMERGENCY ALARM !!"
    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.85, 2)
    x = (w - tw) // 2
    cv2.rectangle(frame, (x - 10, 40), (x + tw + 10, 40 + th + 18),
                  (0, 0, 180), -1)
    cv2.putText(frame, msg, (x, 40 + th + 4),
                FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def draw_stats(frame, h, w, in_c, out_c, ppl_c):
    """
    Compact bold stats bar at bottom.
    Format:  PEOPLE: 5  |  IN: 3  |  OUT: 2
    """
    panel_h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    cv2.line(frame, (0, h - panel_h), (w, h - panel_h), (40, 40, 60), 2)

    scale = 0.72
    thick = 2
    sep   = "  |  "
    pieces = [
        ("PEOPLE: ", (220, 220,  60)),
        (str(ppl_c), (255, 230,   0)),
        (sep,        ( 50,  50,  80)),
        ("IN: ",     ( 60, 220,  60)),
        (str(in_c),  (  0, 255,  80)),
        (sep,        ( 50,  50,  80)),
        ("OUT: ",    ( 80,  80, 255)),
        (str(out_c), ( 60,  80, 255)),
    ]

    total_w = sum(cv2.getTextSize(t, FONT, scale, thick)[0][0] for t, _ in pieces)
    cx = (w - total_w) // 2
    th = cv2.getTextSize("A", FONT, scale, thick)[0][1]
    y  = h - panel_h + th + 12

    for txt, color in pieces:
        cv2.putText(frame, txt, (cx, y), FONT, scale, color, thick, cv2.LINE_AA)
        cx += cv2.getTextSize(txt, FONT, scale, thick)[0][0]


def draw_hints(frame, h, w):
    """Keys in black box — top-right corner, readable size."""
    lines = [
        "F = Full Screen",
        "A = Alarm",
        "S = Stop",
        "R = Reset",
        "Q = Quit",
    ]
    scale  = 0.42
    thick  = 1
    pad    = 7
    line_h = 22

    # Measure widest line for box width
    max_tw = max(cv2.getTextSize(t, FONT, scale, thick)[0][0] for t in lines)
    box_w  = max_tw + pad * 2
    box_h  = line_h * len(lines) + pad * 2
    bx     = w - box_w - 5   # right side
    by     = 5                # top side

    # Black box background
    cv2.rectangle(frame, (bx - pad, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (bx - pad, by), (bx + box_w, by + box_h), (50, 50, 50), 1)

    for i, txt in enumerate(lines):
        y = by + pad + (i + 1) * line_h - 4
        cv2.putText(frame, txt, (bx, y), FONT, scale,
                    (200, 200, 200), thick, cv2.LINE_AA)


def draw_top_labels(frame, line_x, in_c, out_c):
    """Bold IN / OUT labels at the top of the counting line."""
    scale = 0.70
    thick = 2
    pad   = 6

    # IN — right side of line, green
    in_txt = "IN"
    (tw, th), _ = cv2.getTextSize(in_txt, FONT, scale, thick)
    ix = line_x + 10
    iy = th + pad + 5
    cv2.rectangle(frame, (ix - pad, 5), (ix + tw + pad, iy + pad), (0, 0, 0), -1)
    cv2.putText(frame, in_txt, (ix, iy), FONT, scale, (0, 255, 80), thick, cv2.LINE_AA)

    # OUT — left side of line, blue
    out_txt = "OUT"
    (tw2, _), _ = cv2.getTextSize(out_txt, FONT, scale, thick)
    ox = line_x - tw2 - 10
    cv2.rectangle(frame, (ox - pad, 5), (ox + tw2 + pad, iy + pad), (0, 0, 0), -1)
    cv2.putText(frame, out_txt, (ox, iy), FONT, scale, (80, 100, 255), thick, cv2.LINE_AA)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                        help="Camera index (0,1,2...) or video file path")
    parser.add_argument("--notify", default="",
                        help="ntfy.sh topic for phone notification (free)")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    notify_topic = args.notify.strip()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    WIN = "People Counting System"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, args.width, args.height)

    tracker    = Tracker(iou_thresh=0.20, max_missed=25)
    in_count   = 0
    out_count  = 0
    fullscreen = False
    alarm_on   = False

    print(f"\n[START] Camera opened.")
    print(f"[START] Notify topic: '{notify_topic or 'NOT SET — use --notify TOPIC'}'")
    print("[KEYS]  F=Fullscreen  R=Reset  A=Alarm  S=Stop  Q=Quit\n")

    # ── Startup test notification ──────────────────────────────────────────────
    test_notification(notify_topic)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # Resize to consistent display size
        frame = cv2.resize(frame, (args.width, args.height))
        h, w  = frame.shape[:2]

        stats_bar_h = 42
        video_h     = h - stats_bar_h
        line_x      = w // 2

        # ── Detect ────────────────────────────────────────────────────────
        video_region = frame[:video_h, :]
        dets         = detect_persons(video_region)

        # ── Track ─────────────────────────────────────────────────────────
        active_tracks = tracker.update(dets)

        # ── Count crossings ────────────────────────────────────────────────
        for trk in active_tracks:
            cx, cy = trk.centroid
            new_side = "right" if cx >= line_x else "left"

            if trk.side is None:
                trk.side = new_side
            elif trk.side != new_side:
                if trk.side == "left" and new_side == "right":
                    in_count += 1
                    print(f"[COUNT] ID:{trk.id} → IN  | in={in_count}")
                elif trk.side == "right" and new_side == "left":
                    out_count += 1
                    print(f"[COUNT] ID:{trk.id} → OUT | out={out_count}")
                trk.side = new_side

        # Total unique people ever detected (all IDs generated so far)
        people_count = in_count + out_count

        # ── Draw counting line (labels near bottom) ───────────────────────
        draw_counting_line(frame, line_x, video_h)

        # ── Draw bounding boxes + unique IDs ──────────────────────────────
        for trk in active_tracks:
            x1, y1, x2, y2 = trk.bbox
            cx, cy = trk.centroid

            # Green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)

            # ID label above box — black text on green bg
            id_txt = f"ID:{trk.id}"
            (lw, lh), _ = cv2.getTextSize(id_txt, FONT, 0.58, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 8, y1),
                          (0, 180, 0), -1)
            cv2.putText(frame, id_txt, (x1 + 4, y1 - 4),
                        FONT, 0.58, (0, 0, 0), 1, cv2.LINE_AA)

            # Small centroid dot
            cv2.circle(frame, (cx, cy), 4, (0, 255, 150), -1)

        # ── Top IN / OUT bold labels (left & right of line, top) ─────────────
        draw_top_labels(frame, line_x, in_count, out_count)

        # ── Bottom stats: PEOPLE | IN | OUT ───────────────────────────────
        draw_stats(frame, h, w, in_count, out_count, people_count)

        # ── Emergency overlay (original — unchanged) ───────────────────────
        if alarm_on:
            draw_alarm_overlay(frame, video_h, w)

        # ── Hints — black box, top-right corner ───────────────────────────
        draw_hints(frame, h, w)

        cv2.imshow(WIN, frame)

        # ── Keys ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('f') or key == ord('F'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        elif key == ord('r') or key == ord('R'):
            in_count  = 0
            out_count = 0
            tracker   = Tracker(iou_thresh=0.20, max_missed=25)
            print("[RESET] All counts reset to 0.")

        elif key == ord('a') or key == ord('A'):
            alarm_on = True
            start_alarm()
            msg = f"🚨 EMERGENCY ALERT! | PEOPLE(In+Out): {people_count} | IN: {in_count} OUT: {out_count}"
            send_notification_async(notify_topic, msg)
            print(f"[ALARM] Emergency alarm TRIGGERED! Notification phone par bhej raha hai...")
            print(f"[ALARM] Message: {msg}")

        elif key == ord('s') or key == ord('S'):
            if alarm_on:
                alarm_on = False
                stop_alarm()
                print("[ALARM] Emergency alarm stopped.")

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    print("[EXIT] Done.")


if __name__ == "__main__":
    main()
