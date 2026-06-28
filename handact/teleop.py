"""
MediaPipe hand-tracking teleoperation interface.

Maps webcam hand landmarks to a 6-DOF end-effector delta that drives the
gym-aloha simulated arm.  No extra hardware required — just a webcam.

Coordinate convention (matches gym-aloha / MuJoCo world frame):
  x → right,  y → up,  z → toward camera
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TeleopConfig:
    # Scaling: how much a hand movement (normalised [0,1]) maps to robot delta
    pos_scale: float = 0.05       # metres per unit of normalised hand movement
    rot_scale: float = 0.8        # radians per unit of wrist rotation proxy

    # Exponential moving average smoothing (closer to 1 → more smoothing)
    smooth_alpha: float = 0.4

    # Gripper: pinch distance thresholds (normalised)
    gripper_open_thresh: float = 0.08
    gripper_close_thresh: float = 0.04

    # Webcam device index
    camera_index: int = 0

    # Mirror the feed so it feels natural
    mirror: bool = True

    # MediaPipe detection confidence
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


@dataclass
class TeleopState:
    """Smoothed, ready-to-use delta action at each timestep."""
    dx: float = 0.0   # end-effector position delta x
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0
    dpitch: float = 0.0
    dyaw: float = 0.0
    gripper: float = 1.0   # 1.0 = open, 0.0 = closed
    hand_visible: bool = False


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class HandTeleop:
    """
    Reads one frame from the webcam, detects hand landmarks with MediaPipe,
    and returns a TeleopState with smoothed 6-DOF deltas.

    Usage
    -----
    teleop = HandTeleop()
    teleop.start()
    try:
        while collecting:
            state = teleop.step()
            action = state_to_action(state)
    finally:
        teleop.stop()
    """

    def __init__(self, config: Optional[TeleopConfig] = None):
        self.cfg = config or TeleopConfig()
        self._hands = None
        self._cap = None
        self._prev_state = TeleopState()

        # Origin reference: set on first detection so deltas start at zero
        self._origin: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self._cap = cv2.VideoCapture(self.cfg.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cfg.camera_index}")

    def stop(self):
        if self._hands:
            self._hands.close()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self) -> TeleopState:
        """Process one camera frame and return smoothed TeleopState."""
        ret, frame = self._cap.read()
        if not ret:
            return self._prev_state

        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            state = TeleopState(hand_visible=False)
            self._origin = None   # reset so next appearance starts fresh
            self._prev_state = state
            return state

        lm = results.multi_hand_landmarks[0].landmark

        # -- Position: use wrist (0) as base, index MCP (5) for orientation --
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        index_mcp = np.array([lm[5].x, lm[5].y, lm[5].z])
        middle_mcp = np.array([lm[9].x, lm[9].y, lm[9].z])

        # Initialise origin on first visible frame
        if self._origin is None:
            self._origin = wrist.copy()

        delta_pos = (wrist - self._origin) * self.cfg.pos_scale
        # Flip y: MediaPipe y increases downward, robot y increases upward
        delta_pos[1] *= -1

        # -- Rotation proxy: palm normal approximated via cross product --
        palm_x = index_mcp - wrist
        palm_y = middle_mcp - wrist
        palm_normal = np.cross(palm_x, palm_y)
        norm = np.linalg.norm(palm_normal)
        if norm > 1e-6:
            palm_normal /= norm

        droll  = float(palm_normal[2]) * self.cfg.rot_scale
        dpitch = float(palm_normal[0]) * self.cfg.rot_scale
        dyaw   = float(palm_normal[1]) * self.cfg.rot_scale

        # -- Gripper: pinch distance between thumb tip (4) and index tip (8) --
        thumb_tip = np.array([lm[4].x, lm[4].y, lm[4].z])
        index_tip = np.array([lm[8].x, lm[8].y, lm[8].z])
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

        if pinch_dist < self.cfg.gripper_close_thresh:
            gripper = 0.0
        elif pinch_dist > self.cfg.gripper_open_thresh:
            gripper = 1.0
        else:
            # linear interpolation in the dead-band
            t = (pinch_dist - self.cfg.gripper_close_thresh) / (
                self.cfg.gripper_open_thresh - self.cfg.gripper_close_thresh
            )
            gripper = float(t)

        # -- Exponential smoothing --
        a = self.cfg.smooth_alpha
        p = self._prev_state
        state = TeleopState(
            dx     = a * float(delta_pos[0]) + (1 - a) * p.dx,
            dy     = a * float(delta_pos[1]) + (1 - a) * p.dy,
            dz     = a * float(delta_pos[2]) + (1 - a) * p.dz,
            droll  = a * droll  + (1 - a) * p.droll,
            dpitch = a * dpitch + (1 - a) * p.dpitch,
            dyaw   = a * dyaw   + (1 - a) * p.dyaw,
            gripper = a * gripper + (1 - a) * p.gripper,
            hand_visible=True,
        )

        self._prev_state = state

        # -- Optional live preview --
        self._draw_overlay(frame, lm, state)
        cv2.imshow("HandACT Teleop", frame)
        cv2.waitKey(1)

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_overlay(self, frame, landmarks, state: TeleopState):
        mp_draw = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        # Draw skeleton
        h, w, _ = frame.shape
        # Re-wrap into a proto-compatible object for drawing
        # (we already have the raw landmark list so draw manually)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # HUD
        gripper_pct = int(state.gripper * 100)
        cv2.putText(frame, f"Gripper: {gripper_pct}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"dx={state.dx:.3f} dy={state.dy:.3f} dz={state.dz:.3f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)


# ---------------------------------------------------------------------------
# Conversion helper: TeleopState → flat numpy action vector
# ---------------------------------------------------------------------------

def teleop_state_to_action(state: TeleopState) -> np.ndarray:
    """
    Returns a (7,) float32 array: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    This matches the gym-aloha end-effector delta action space.
    """
    return np.array(
        [state.dx, state.dy, state.dz,
         state.droll, state.dpitch, state.dyaw,
         state.gripper],
        dtype=np.float32,
    )
