"""
generate_alarm.py
=================
Generates a simple alarm.wav beep sound using only numpy and wave.
No external downloads needed.

Run:
    python generate_alarm.py
"""

import wave
import struct
import math
import os


def generate_beep(filename="alarm.wav", frequency=880, duration=1.0,
                  sample_rate=44100, volume=0.6):
    """
    Generate a simple sine wave beep and save as WAV.
    frequency : Hz  (880 = A5, a sharp attention-grabbing tone)
    duration  : seconds
    volume    : 0.0 to 1.0
    """
    n_samples = int(sample_rate * duration)
    max_amplitude = int(32767 * volume)

    with wave.open(filename, "w") as wf:
        wf.setnchannels(1)        # mono
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(sample_rate)

        frames = []
        for i in range(n_samples):
            # Sine wave
            t = i / sample_rate
            sample = max_amplitude * math.sin(2 * math.pi * frequency * t)

            # Short fade in/out to avoid clicking
            fade_samples = int(sample_rate * 0.02)
            if i < fade_samples:
                sample *= i / fade_samples
            elif i > n_samples - fade_samples:
                sample *= (n_samples - i) / fade_samples

            frames.append(struct.pack("<h", int(sample)))

        wf.writeframes(b"".join(frames))

    size_kb = os.path.getsize(filename) / 1024
    print(f"[OK] Generated {filename} ({size_kb:.1f} KB, {frequency}Hz, {duration}s)")


if __name__ == "__main__":
    generate_beep("alarm.wav", frequency=880, duration=1.2)
    print("     You can now run the detector — alarm.wav is ready.")
