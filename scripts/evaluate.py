"""
Evaluate a trained ACT policy in the gym-aloha simulator.

Features:
  - Loads a checkpoint produced by train.py
  - Runs chunk execution: re-queries the policy every `n_action_steps` steps
  - Applies temporal ensembling to smooth transitions between chunks
  - Reports success rate over N rollouts

Usage:
    python scripts/evaluate.py --checkpoint outputs/train/policy_final.pt \
                                --num-episodes 20 \
                                --task gym_aloha/AlohaInsertion-v0
"""

import argparse
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy


# ---------------------------------------------------------------------------
# Temporal ensembling
# ---------------------------------------------------------------------------

class TemporalEnsembler:
    """
    Blends overlapping action chunks via exponential decay weights.

    At each timestep t, multiple chunks may have predicted an action for t.
    The most recent prediction gets weight 1, the previous exp(-k), etc.
    This matches the scheme in the original ACT paper.
    """

    def __init__(self, chunk_size: int, coeff: float = 0.01):
        self.chunk_size = chunk_size
        self.coeff = coeff
        # Buffer of (chunk, age) pairs; age counts steps since the chunk was predicted
        self._chunks: deque[tuple[np.ndarray, int]] = deque()

    def add_chunk(self, chunk: np.ndarray):
        """Register a newly predicted action chunk (shape: [chunk_size, action_dim])."""
        self._chunks.append((chunk.copy(), 0))

    def get_action(self, step_within_chunk: int) -> np.ndarray:
        """Return the blended action for the current timestep."""
        if not self._chunks:
            raise RuntimeError("No chunks registered yet.")

        weights, actions = [], []
        for chunk, age in self._chunks:
            if step_within_chunk < len(chunk):
                actions.append(chunk[step_within_chunk])
                weights.append(np.exp(-self.coeff * age))

        weights = np.array(weights)
        weights /= weights.sum()
        blended = sum(w * a for w, a in zip(weights, actions))

        # Age all chunks
        self._chunks = deque((c, a + 1) for c, a in self._chunks)

        # Evict chunks that have been fully consumed
        while self._chunks and self._chunks[0][1] >= self.chunk_size:
            self._chunks.popleft()

        return blended


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_episode(
    env,
    policy: ACTPolicy,
    ensembler: TemporalEnsembler,
    chunk_size: int,
    n_action_steps: int,
    device: torch.device,
    max_steps: int = 400,
) -> dict:
    obs, _ = env.reset()
    ensembler._chunks.clear()

    total_reward = 0.0
    success = False
    step = 0
    chunk_step = n_action_steps   # force prediction on first timestep

    while step < max_steps:
        # Re-query the policy every n_action_steps
        if chunk_step >= n_action_steps:
            obs_tensor = _obs_to_tensor(obs, device)
            chunk = policy.select_action(obs_tensor)   # (chunk_size, action_dim)
            chunk_np = chunk.cpu().numpy()
            ensembler.add_chunk(chunk_np)
            chunk_step = 0

        action = ensembler.get_action(chunk_step)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            success = info.get("is_success", reward > 0)
            break
        if truncated:
            break

        chunk_step += 1
        step += 1

    return {"success": success, "total_reward": total_reward, "steps": step}


def _obs_to_tensor(obs: dict, device: torch.device) -> dict:
    """Convert a gym observation dict to batched tensors."""
    result = {}
    img = obs["images"]["top"]   # (H, W, 3) uint8
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    result["observation.images.top"] = img_t.unsqueeze(0).to(device)

    state = obs["agent_pos"].astype(np.float32)
    result["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", default="gym_aloha/AlohaInsertion-v0")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=100)
    p.add_argument("--n-action-steps", type=int, default=100)
    p.add_argument("--ensemble-coeff", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--device", default="cuda")
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Load policy --
    cfg = ACTConfig(chunk_size=args.chunk_size, n_action_steps=args.n_action_steps)
    policy = ACTPolicy(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(ckpt)
    policy.to(device)
    policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # -- Env --
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make(args.task, render_mode=render_mode)

    ensembler = TemporalEnsembler(
        chunk_size=args.chunk_size,
        coeff=args.ensemble_coeff,
    )

    # -- Evaluation loop --
    results = []
    for ep in range(args.num_episodes):
        result = run_episode(
            env, policy, ensembler,
            chunk_size=args.chunk_size,
            n_action_steps=args.n_action_steps,
            device=device,
            max_steps=args.max_steps,
        )
        results.append(result)
        status = "SUCCESS" if result["success"] else "fail"
        print(f"  Episode {ep + 1:3d}: {status}  "
              f"reward={result['total_reward']:.2f}  steps={result['steps']}")

    success_rate = np.mean([r["success"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    print(f"\n{'='*40}")
    print(f"Success rate : {success_rate:.1%}  ({int(success_rate * args.num_episodes)}/{args.num_episodes})")
    print(f"Avg reward   : {avg_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
