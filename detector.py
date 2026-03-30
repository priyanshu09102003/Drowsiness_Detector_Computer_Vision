"""
Real-Time Driver Drowsiness Detection System
============================================
Detects drowsiness using Eye Aspect Ratio (EAR) + optional CNN classifier.
Works with webcam (live) or a video file (headless/server mode).

Usage:
    # Webcam mode (with GUI)
    python detector.py

    # Headless mode with video file (server/evaluator friendly)
    python detector.py --input sample_video.mp4 --output result.avi --headless

    # Webcam with output saved
    python detector.py --output result.avi
"""

import cv2
import dlib
import numpy as np
import time
import threading
import argparse
import os
import sys
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils

# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Driver Drowsiness Detection System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument(
        "-i", "--input", default=None,
        help="Path to input video file.\n"
             "If omitted, webcam (src=0) is used."
    )
    ap.add_argument(
        "-o", "--output", default=None,
        help="Path to save annotated output video (e.g. result.avi).\n"
             "Optional — omit if you do not need a saved copy."
    )
    ap.add_argument(
        "--headless", action="store_true",
        help="Run without GUI window (required on servers without a display)."
    )
    ap.add_argument(
        "--ear-thresh", type=float, default=0.25,
        help="Eye Aspect Ratio threshold (default: 0.25)."
    )
    ap.add_argument(
        "--ear-frames", type=int, default=20,
        help="Consecutive frames below EAR threshold before alarm (default: 20)."
    )
    ap.add_argument(
        "--mar-thresh", type=float, default=0.60,
        help="Mouth Aspect Ratio threshold for yawn detection (default: 0.60)."
    )
    ap.add_argument(
        "--mar-frames", type=int, default=15,
        help="Consecutive frames above MAR threshold before yawn alert (default: 15)."
    )
    return vars(ap.parse_args())


# ── EAR / MAR Calculations ────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    """
    Eye Aspect Ratio (EAR):
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Drops sharply when the eye closes.
    Reference: Soukupova & Cech, 2016.
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    """
    Mouth Aspect Ratio (MAR):
    Ratio of vertical mouth opening to horizontal width.
    Rises when the driver yawns.
    """
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)


# ── CNN Classifier (optional) ─────────────────────────────────────────────────

def load_cnn_model(model_path="model/eye_model.h5"):
    """Load pre-trained CNN if available. Returns None otherwise."""
    if not os.path.exists(model_path):
        return None
    try:
        from tensorflow.keras.models import load_model
        print(f"[INFO] CNN model loaded from {model_path}")
        return load_model(model_path)
    except Exception as e:
        print(f"[WARN] Could not load CNN model: {e}")
        return None


def cnn_predict_eye(model, eye_roi):
    """Run CNN on a cropped eye ROI. Returns probability (1.0=open, 0.0=closed)."""
    if model is None or eye_roi is None or eye_roi.size == 0:
        return None
    try:
        roi = cv2.resize(eye_roi, (24, 24))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))
        return float(model.predict(roi, verbose=0)[0][0])
    except Exception:
        return None


# ── Alarm ─────────────────────────────────────────────────────────────────────

def play_alarm(alarm_path="alarm.wav"):
    """Play alarm sound in a daemon thread. Falls back to system beep."""
    def _play():
        if os.path.exists(alarm_path):
            try:
                from playsound import playsound
                playsound(alarm_path)
                return
            except Exception:
                pass
        sys.stdout.write("\a")
        sys.stdout.flush()

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ── Overlay / HUD ─────────────────────────────────────────────────────────────

def draw_hud(frame, ear, mar, blink_count, yawn_count,
             drowsy, yawning, cnn_state=None):
    """Draw semi-transparent HUD bar and warning overlays onto frame."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    white  = (255, 255, 255)
    gray   = (170, 170, 170)
    orange = (0, 165, 255)
    red    = (0, 0, 255)
    green  = (0, 220, 90)

    cv2.putText(frame, f"EAR: {ear:.3f}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2)
    cv2.putText(frame, f"MAR: {mar:.3f}", (170, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, white, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gray, 1)
    cv2.putText(frame, f"Yawns: {yawn_count}", (160, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gray, 1)

    if cnn_state is not None:
        label = f"CNN: {'Open' if cnn_state > 0.5 else 'Closed'} ({cnn_state:.2f})"
        cv2.putText(frame, label, (320, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1)

    if drowsy:
        cv2.rectangle(frame, (0, 0), (w, h), red, 8)
        cv2.putText(frame, "DROWSINESS ALERT!", (w // 2 - 190, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.3, red, 3)

    if yawning:
        cv2.putText(frame, "YAWNING DETECTED", (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, orange, 2)

    return frame


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    EAR_THRESHOLD     = args["ear_thresh"]
    EAR_CONSEC_FRAMES = args["ear_frames"]
    MAR_THRESHOLD     = args["mar_thresh"]
    MAR_CONSEC_FRAMES = args["mar_frames"]

    LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(LANDMARK_MODEL):
        print(f"[ERROR] Landmark model not found: {LANDMARK_MODEL}")
        print("  Run: python setup.py")
        print("  Or download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    print("[INFO] Loading dlib models...")
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(LANDMARK_MODEL)
    cnn_model = load_cnn_model()

    (lS, lE) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mS, mE) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

    src = args["input"] if args["input"] else 0
    print(f"[INFO] Opening video source: {'webcam' if src == 0 else src}")
    vs = cv2.VideoCapture(src)

    if not vs.isOpened():
        print("[ERROR] Could not open video source.")
        sys.exit(1)

    writer = None
    if args["output"]:
        W   = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vs.get(cv2.CAP_PROP_FPS) or 20
        new_h = int(H * 720 / W)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, fps, (720, new_h))
        print(f"[INFO] Writing output to: {args['output']}")

    eye_counter  = 0
    yawn_counter = 0
    blink_total  = 0
    yawn_total   = 0
    alarm_on     = False
    frame_num    = 0
    start_time   = time.time()

    print("[INFO] Processing... Press 'q' to quit (GUI mode only).")
    print(f"       EAR threshold={EAR_THRESHOLD}, frames={EAR_CONSEC_FRAMES}")
    print(f"       MAR threshold={MAR_THRESHOLD}, frames={MAR_CONSEC_FRAMES}")

    while True:
        ret, frame = vs.read()
        if not ret or frame is None:
            print("[INFO] End of video stream.")
            break

        frame     = imutils.resize(frame, width=720)
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_num += 1

        faces     = detector(gray, 0)
        ear       = 0.0
        mar       = 0.0
        cnn_state = None

        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye  = shape[lS:lE]
            right_eye = shape[rS:rE]
            mouth     = shape[mS:mE]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear       = (left_ear + right_ear) / 2.0
            mar       = mouth_aspect_ratio(mouth)

            for pts in [left_eye, right_eye]:
                hull = cv2.convexHull(pts)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 165, 255), 1)

            if cnn_model is not None:
                (x1, y1, x2, y2) = cv2.boundingRect(left_eye)
                pad = 4
                eye_roi   = gray[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
                cnn_state = cnn_predict_eye(cnn_model, eye_roi)

            # EAR drowsiness logic
            if ear < EAR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= EAR_CONSEC_FRAMES and not alarm_on:
                    alarm_on = True
                    play_alarm()
            else:
                if eye_counter >= EAR_CONSEC_FRAMES:
                    blink_total += 1
                eye_counter = 0
                alarm_on    = False

            # MAR yawn logic
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter == MAR_CONSEC_FRAMES:
                    yawn_total += 1
            else:
                yawn_counter = 0

        drowsy  = alarm_on
        yawning = yawn_counter >= MAR_CONSEC_FRAMES

        frame = draw_hud(frame, ear, mar, blink_total, yawn_total,
                         drowsy, yawning, cnn_state)

        if writer:
            writer.write(frame)

        if not args["headless"]:
            cv2.imshow("Drowsiness Detector — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args["headless"] and frame_num % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [Frame {frame_num}] EAR={ear:.3f} MAR={mar:.3f} "
                  f"Blinks={blink_total} Yawns={yawn_total} "
                  f"Elapsed={elapsed:.1f}s")

    elapsed = time.time() - start_time
    vs.release()
    if writer:
        writer.release()
    if not args["headless"]:
        cv2.destroyAllWindows()

    print("\n[DONE] Summary")
    print(f"  Frames processed : {frame_num}")
    print(f"  Total blinks     : {blink_total}")
    print(f"  Total yawns      : {yawn_total}")
    print(f"  Time elapsed     : {elapsed:.2f}s")
    if args["output"]:
        print(f"  Output saved to  : {args['output']}")


if __name__ == "__main__":
    main()
