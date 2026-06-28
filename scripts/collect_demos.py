"""
Demo collection: hand-tracking teleop → LeRobot dataset.

Run:
    python scripts/collect_demos.py --num-episodes 50 --task gym_aloha/AlohaInsertion-v0

Controls (shown in the OpenCV preview window):
    SPACE  – start / stop recording an episode
    R      – discard the current episode and start over
    Q      – quit and save the dataset
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# LeRobot dataset API (lerobot >= 0.3)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from handact.teleop import HandTeleop, TeleopConfig, teleop_state_to_action


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(task: str):
    """Return a reset-able gym-aloha environment."""
    env = gym.make(task, render_mode="human")
    return env


def collect_episode(env, teleop: HandTeleop, fps: int = 50) -> list[dict]:
    """
    Run one teleoperated episode.  Returns a list of transition dicts
    (each dict has keys: observation, action, reward, done).
    """
    obs, _ = env.reset()
    transitions = []
    dt = 1.0 / fps

    print("  Episode started — move your hand to control the arm.")
    while True:
        t0 = time.perf_counter()

        state = teleop.step()
        action = teleop_state_to_action(state)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        transitions.append({
            "observation": obs,
            "action": action,
            "reward": float(reward),
            "done": done,
        })

        obs = next_obs

        # Hold frame rate
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, dt - elapsed))

        if done:
            break

    print(f"  Episode done — {len(transitions)} steps, reward={reward:.2f}")
    return transitions


def transitions_to_lerobot(
    transitions: list[dict],
    dataset: LeRobotDataset,
    episode_index: int,
    task: str,
):
    """Push one episode of transitions into a LeRobotDataset."""
    for step_idx, t in enumerate(transitions):
        obs = t["observation"]

        # gym-aloha returns a dict with image + agent_pos keys
        frame = {
            "observation.images.top": obs["images"]["top"],          # (H, W, 3) uint8
            "observation.state": obs["agent_pos"].astype(np.float32),
            "action": t["action"],
            "episode_index": episode_index,
            "frame_index": step_idx,
            "timestamp": step_idx / 50.0,
            "next.reward": t["reward"],
            "next.done": t["done"],
            "task_index": 0,
        }
        dataset.add_frame(frame)

    dataset.save_episode(task=task)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="gym_aloha/AlohaInsertion-v0",
                   help="Gymnasium task id")
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/demos"))
    p.add_argument("--repo-id", default=None,
                   help="HuggingFace Hub repo id to push to (optional)")
    p.add_argument("--camera-index", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Task          : {args.task}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Output dir    : {args.output_dir}")
    print()

    # ------------------------------------------------------------------
    # Initialise LeRobot dataset
    # ------------------------------------------------------------------
    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),   # 14 joint positions for bimanual ALOHA
            "names": ["motor"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action_dim"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id or "local/handact-demos",
        fps=args.fps,
        features=features,
        root=args.output_dir,
    )

    # ------------------------------------------------------------------
    # Initialise teleop + environment
    # ------------------------------------------------------------------
    teleop_cfg = TeleopConfig(camera_index=args.camera_index)
    teleop = HandTeleop(config=teleop_cfg)
    teleop.start()

    env = make_env(args.task)

    # ------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------
    episode = 0
    try:
        while episode < args.num_episodes:
            print(f"\n[Episode {episode + 1}/{args.num_episodes}]")
            print("  Press SPACE in the teleop window to start, R to retry, Q to quit.")

            transitions = collect_episode(env, teleop, fps=args.fps)

            if len(transitions) < 10:
                print("  Too short — discarding.")
                continue

            transitions_to_lerobot(transitions, dataset, episode, task=args.task)
            episode += 1
            print(f"  Saved. ({episode}/{args.num_episodes} episodes complete)")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        teleop.stop()
        env.close()

    # ------------------------------------------------------------------
    # Optional Hub upload
    # ------------------------------------------------------------------
    if args.repo_id:
        print(f"\nPushing dataset to Hub: {args.repo_id}")
        dataset.push_to_hub()
        print("Done.")
    else:
        print(f"\nDataset saved locally at: {args.output_dir}")
        print("Re-run with --repo-id <your-hf-username>/handact-demos to publish.")


if __name__ == "__main__":
    main()
