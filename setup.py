"""
setup.py — One-time environment setup for Drowsiness Detector
=============================================================
Run this ONCE before using the project:
    python setup.py

What it does:
  1. Checks Python version (>= 3.8 required)
  2. Installs all pip dependencies
  3. Downloads the dlib 68-point facial landmark model
  4. Creates required empty directories
  5. Runs a quick sanity check
"""

import sys
import os
import subprocess
import urllib.request


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 8):
        print(f"[ERROR] Python 3.8+ required. You have {v.major}.{v.minor}")
        sys.exit(1)
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")


def install_requirements():
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print("[INFO] Installing requirements (this may take a few minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req, "--quiet"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[WARN] Some packages may have failed. Error output:")
        print(result.stderr[-2000:])
    else:
        print("[OK] Requirements installed.")


def download_landmark_model():
    dat_file = "shape_predictor_68_face_landmarks.dat"
    bz2_file = dat_file + ".bz2"
    url = f"http://dlib.net/files/{bz2_file}"

    if os.path.exists(dat_file):
        size_mb = os.path.getsize(dat_file) / 1e6
        print(f"[OK] {dat_file} already exists ({size_mb:.1f} MB) — skipping download.")
        return

    print(f"[INFO] Downloading dlib landmark model (~99 MB)...")
    print(f"       Source: {url}")
    print("       This may take a few minutes on a slow connection...")

    try:
        urllib.request.urlretrieve(url, bz2_file, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("  Please download manually:")
        print(f"  1. Go to: {url}")
        print(f"  2. Save the .bz2 file to this folder")
        print(f"  3. Run:  bzip2 -dk {bz2_file}")
        return

    print("[INFO] Extracting .bz2 archive...")
    try:
        import bz2
        with bz2.open(bz2_file, "rb") as f_in:
            data = f_in.read()
        with open(dat_file, "wb") as f_out:
            f_out.write(data)
        os.remove(bz2_file)
        print(f"[OK] {dat_file} ready.")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        print(f"  Try manually: bzip2 -dk {bz2_file}")


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(100, count * block_size * 100 // total_size)
    bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
    print(f"\r  [{bar}] {pct}%", end="", flush=True)


def create_dirs():
    dirs = [
        "model",
        "haarcascades",
        os.path.join("dataset", "open"),
        os.path.join("dataset", "closed"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Directory structure verified.")


def generate_alarm():
    """Auto-generate alarm.wav if it doesn't exist."""
    if os.path.exists("alarm.wav"):
        print("[OK] alarm.wav already exists — skipping.")
        return
    if os.path.exists("generate_alarm.py"):
        print("[INFO] Generating alarm.wav...")
        result = subprocess.run(
            [sys.executable, "generate_alarm.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[OK] alarm.wav generated.")
        else:
            print("[WARN] Could not generate alarm.wav — system beep will be used as fallback.")
    else:
        print("[WARN] generate_alarm.py not found — system beep will be used as fallback.")


def sanity_check():
    print("\n[INFO] Running sanity checks...")

    required = {
        "cv2":     "opencv-python",
        "dlib":    "dlib",
        "imutils": "imutils",
        "scipy":   "scipy",
        "numpy":   "numpy",
    }

    all_ok = True
    for module, pkg in required.items():
        try:
            __import__(module)
            print(f"  [OK]   {pkg}")
        except ImportError:
            print(f"  [FAIL] {pkg} — run: pip install {pkg}")
            all_ok = False

    try:
        import tensorflow as tf
        print(f"  [OK]   tensorflow ({tf.__version__})")
    except ImportError:
        print("  [WARN] tensorflow not installed — CNN features will be disabled")
        print("         Install with: pip install tensorflow")

    dat = "shape_predictor_68_face_landmarks.dat"
    if os.path.exists(dat):
        size_mb = os.path.getsize(dat) / 1e6
        print(f"  [OK]   {dat} ({size_mb:.1f} MB)")
    else:
        print(f"  [FAIL] {dat} not found")
        all_ok = False

    if os.path.exists("alarm.wav"):
        print("  [OK]   alarm.wav")
    else:
        print("  [WARN] alarm.wav not found — system beep fallback will be used")

    return all_ok


def main():
    print("=" * 58)
    print("  Drowsiness Detector — One-Time Setup")
    print("=" * 58)

    check_python()
    create_dirs()
    install_requirements()
    download_landmark_model()
    generate_alarm()
    ok = sanity_check()

    print("\n" + "=" * 58)
    if ok:
        print("  Setup complete!")
        print()
        print("  QUICK START (headless / evaluator mode):")
        print()
        print("    Step 1 — Generate sample test video:")
        print("      python generate_sample_video.py")
        print()
        print("    Step 2 — Run detector:")
        print("      python detector.py --input sample_video.mp4 \\")
        print("                         --output result.avi --headless")
        print()
        print("  LIVE WEBCAM mode (requires display):")
        print("      python detector.py")
        print()
        print("  OPTIONAL — Train CNN classifier:")
        print("      python train_cnn.py")
    else:
        print("  Setup finished with errors. Fix the [FAIL] items above,")
        print("  then re-run: python setup.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
