"""
generate_sample_video.py
========================
Generates a synthetic test video (sample_video.mp4) for headless evaluation.
No webcam or real face needed — useful for evaluators running on a server.

Run once:
    python generate_sample_video.py

Then test the detector:
    python detector.py --input sample_video.mp4 --output result.avi --headless
"""

import cv2
import numpy as np
import os


def draw_face(frame, t, phase):
    """Draw a cartoon face onto frame. Phase controls eye/mouth state."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Face
    cv2.ellipse(frame, (cx, cy), (130, 155), 0, 0, 360, (210, 185, 155), -1)
    cv2.ellipse(frame, (cx, cy), (130, 155), 0, 0, 360, (160, 130, 100), 2)

    # Eyebrows
    cv2.line(frame, (cx - 75, cy - 65), (cx - 35, cy - 58), (80, 60, 40), 3)
    cv2.line(frame, (cx + 35, cy - 58), (cx + 75, cy - 65), (80, 60, 40), 3)

    # Eyes
    if phase == "normal":
        cv2.ellipse(frame, (cx - 55, cy - 30), (28, 18), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + 55, cy - 30), (28, 18), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (cx - 55, cy - 30), 12, (60, 40, 20), -1)
        cv2.circle(frame, (cx + 55, cy - 30), 12, (60, 40, 20), -1)
        cv2.circle(frame, (cx - 50, cy - 35), 4,  (255, 255, 255), -1)
        cv2.circle(frame, (cx + 60, cy - 35), 4,  (255, 255, 255), -1)
    elif phase == "closing":
        eye_h = max(2, int(18 * (0.4 - (t % 0.4) / 0.4 * 0.35)))
        cv2.ellipse(frame, (cx - 55, cy - 30), (28, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (cx + 55, cy - 30), (28, max(2, eye_h)), 0, 0, 360, (255, 255, 255), -1)
        cv2.line(frame, (cx - 83, cy - 30), (cx - 27, cy - 30), (210, 185, 155), 5)
        cv2.line(frame, (cx + 27, cy - 30), (cx + 83, cy - 30), (210, 185, 155), 5)
    else:  # closed
        cv2.ellipse(frame, (cx - 55, cy - 30), (28, 3), 0, 0, 360, (210, 185, 155), -1)
        cv2.ellipse(frame, (cx + 55, cy - 30), (28, 3), 0, 0, 360, (210, 185, 155), -1)
        cv2.line(frame, (cx - 83, cy - 30), (cx - 27, cy - 30), (140, 110, 85), 2)
        cv2.line(frame, (cx + 27, cy - 30), (cx + 83, cy - 30), (140, 110, 85), 2)

    # Nose
    pts = np.array([[cx, cy + 10], [cx - 18, cy + 55], [cx + 18, cy + 55]], np.int32)
    cv2.polylines(frame, [pts], False, (160, 130, 100), 2)

    # Mouth
    if phase == "yawning":
        cv2.ellipse(frame, (cx, cy + 80), (50, 35), 0, 0, 180, (60, 20, 20), -1)
        cv2.ellipse(frame, (cx, cy + 80), (50, 35), 0, 0, 180, (120, 60, 60), 2)
        cv2.rectangle(frame, (cx - 35, cy + 80), (cx + 35, cy + 100), (235, 235, 235), -1)
    else:
        cv2.ellipse(frame, (cx, cy + 80), (40, 18), 0, 0, 180, (150, 75, 75), 2)

    # Status label
    labels = {
        "normal":  ("Normal — eyes open",    (0, 200, 80)),
        "closing": ("Eyes closing...",        (0, 165, 255)),
        "closed":  ("DROWSY — eyes closed",   (0, 60, 220)),
        "yawning": ("Yawning detected",       (0, 120, 255)),
    }
    text, color = labels.get(phase, ("", (200, 200, 200)))
    cv2.putText(frame, text, (w // 2 - 160, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)
    cv2.putText(frame, f"t={t:.1f}s", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1)
    return frame


def main():
    output  = "sample_video.mp4"
    fps     = 20
    duration = 30
    total   = fps * duration
    W, H    = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (W, H))

    print(f"[INFO] Generating {duration}s synthetic test video...")

    for i in range(total):
        t = i / fps

        # Background
        frame = np.full((H, W, 3), (48, 52, 58), dtype=np.uint8)
        for gx in range(0, W, 40):
            cv2.line(frame, (gx, 0), (gx, H), (53, 57, 63), 1)
        for gy in range(0, H, 40):
            cv2.line(frame, (0, gy), (W, gy), (53, 57, 63), 1)

        # Phase schedule:
        # 0–7s   normal
        # 7–9s   closing
        # 9–15s  closed  (triggers drowsiness alert)
        # 15–20s normal
        # 20–25s yawning
        # 25–30s normal
        if   t < 7:   phase = "normal"
        elif t < 9:   phase = "closing"
        elif t < 15:  phase = "closed"
        elif t < 20:  phase = "normal"
        elif t < 25:  phase = "yawning"
        else:         phase = "normal"

        frame = draw_face(frame, t, phase)
        writer.write(frame)

        if i % (fps * 5) == 0:
            print(f"  {t:4.0f}s / {duration}s  [{phase}]")

    writer.release()
    size_mb = os.path.getsize(output) / 1e6
    print(f"\n[DONE] {output} created ({size_mb:.1f} MB)")
    print("\n  Test with:")
    print("    python detector.py --input sample_video.mp4 \\")
    print("                       --output result.avi --headless")


if __name__ == "__main__":
    main()
