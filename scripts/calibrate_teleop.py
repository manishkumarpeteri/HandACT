"""
Interactive calibration tool for HandTeleop parameters.

Run this before collecting demos to dial in the feel of the hand-tracking
control. Adjust sliders in the OpenCV window; the live readout shows what
action values your hand is producing in real time. Press S to save the
tuned config to configs/teleop.yaml.

Usage:
    python scripts/calibrate_teleop.py
    python scripts/calibrate_teleop.py --camera-index 1
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from handact.teleop import HandTeleop, TeleopConfig, teleop_state_to_action


# ---------------------------------------------------------------------------
# Slider helpers (OpenCV trackbars work with integers, so we scale floats)
# ---------------------------------------------------------------------------

WINDOW = "HandACT Calibration"

PARAMS = [
    # (display_name, attr_name, min_val, max_val, scale)
    # scale: slider integer = value * scale
    ("Pos scale   (x100)", "pos_scale",    1,  50, 100),
    ("Rot scale   (x10)",  "rot_scale",    1,  50,  10),
    ("Smoothing   (x100)", "smooth_alpha",  5,  95, 100),
    ("Gripper open thresh (x1000)", "gripper_open_thresh",  10, 200, 1000),
    ("Gripper close thresh (x1000)","gripper_close_thresh",  5, 100, 1000),
]


def make_window(cfg: TeleopConfig):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 900, 600)
    for label, attr, lo, hi, scale in PARAMS:
        val = int(getattr(cfg, attr) * scale)
        cv2.createTrackbar(label, WINDOW, val, hi, lambda x: None)
        # Clamp to [lo, hi] initial value
        cv2.setTrackbarMin(label, WINDOW, lo)


def read_sliders(cfg: TeleopConfig):
    """Pull current slider positions back into cfg (mutates in place)."""
    for label, attr, lo, hi, scale in PARAMS:
        raw = cv2.getTrackbarPos(label, WINDOW)
        setattr(cfg, attr, raw / scale)


# ---------------------------------------------------------------------------
# HUD drawing
# ---------------------------------------------------------------------------

BAR_W = 300
BAR_H = 18
BAR_PAD = 6

def draw_bar(img, x, y, value, lo, hi, label, color):
    """Draw a labelled horizontal bar for a scalar value."""
    frac = np.clip((value - lo) / (hi - lo), 0, 1)
    filled = int(frac * BAR_W)
    cv2.rectangle(img, (x, y), (x + BAR_W, y + BAR_H), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + filled, y + BAR_H), color, -1)
    cv2.putText(img, f"{label}: {value:+.3f}",
                (x + BAR_W + BAR_PAD, y + BAR_H - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def draw_hud(frame, state, action):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent panel
    panel_h = 320
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y0 = h - panel_h + 20
    lx = 20

    cv2.putText(frame, "LIVE ACTION OUTPUT", (lx, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2)
    y0 += 30

    dims = [
        ("dx   (right+)", action[0], -0.15, 0.15, (100, 180, 255)),
        ("dy   (up+)",    action[1], -0.15, 0.15, (100, 180, 255)),
        ("dz   (fwd+)",   action[2], -0.15, 0.15, (100, 180, 255)),
        ("roll",          action[3], -1.0,  1.0,  (200, 140, 100)),
        ("pitch",         action[4], -1.0,  1.0,  (200, 140, 100)),
        ("yaw",           action[5], -1.0,  1.0,  (200, 140, 100)),
        ("gripper",       action[6],  0.0,  1.0,  (100, 220, 100)),
    ]

    for label, val, lo, hi, color in dims:
        draw_bar(frame, lx, y0, val, lo, hi, label, color)
        y0 += BAR_H + BAR_PAD

    # Hand visibility indicator
    status = "HAND DETECTED" if state.hand_visible else "no hand"
    color = (0, 255, 0) if state.hand_visible else (0, 80, 200)
    cv2.putText(frame, status, (lx, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Controls hint
    cv2.putText(frame, "S = save config   Q = quit",
                (w - 280, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ---------------------------------------------------------------------------
# Save config
# ---------------------------------------------------------------------------

def save_config(cfg: TeleopConfig, path: Path):
    data = {
        "pos_scale":            cfg.pos_scale,
        "rot_scale":            cfg.rot_scale,
        "smooth_alpha":         cfg.smooth_alpha,
        "gripper_open_thresh":  cfg.gripper_open_thresh,
        "gripper_close_thresh": cfg.gripper_close_thresh,
        "camera_index":         cfg.camera_index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Config saved to {path}")


def load_config(path: Path) -> TeleopConfig:
    if not path.exists():
        return TeleopConfig()
    with open(path) as f:
        data = yaml.safe_load(f)
    return TeleopConfig(**data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--config", type=Path, default=Path("configs/teleop.yaml"))
    return p.parse_args()


def main():
    args = parse_args()

    # Load existing config if present, else defaults
    cfg = load_config(args.config)
    cfg.camera_index = args.camera_index

    print("Calibration tool starting.")
    print("  Move your hand to see the action bars respond.")
    print("  Adjust sliders to tune feel, then press S to save.\n")

    make_window(cfg)

    teleop = HandTeleop(config=cfg)
    teleop.start()

    try:
        while True:
            # Pull slider values into cfg so HandTeleop picks them up live
            read_sliders(cfg)

            state = teleop.step()
            action = teleop_state_to_action(state)

            # Grab the frame that teleop already displayed and annotate it
            ret, raw_frame = teleop._cap.read()
            if not ret:
                continue
            if cfg.mirror:
                raw_frame = cv2.flip(raw_frame, 1)

            draw_hud(raw_frame, state, action)
            cv2.imshow(WINDOW, raw_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit.")
                break
            elif key == ord("s"):
                save_config(cfg, args.config)
                # Flash confirmation on screen
                confirm = raw_frame.copy()
                cv2.putText(confirm, "SAVED!", (raw_frame.shape[1] // 2 - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)
                cv2.imshow(WINDOW, confirm)
                cv2.waitKey(800)

    finally:
        teleop.stop()


if __name__ == "__main__":
    main()
