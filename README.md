# HandACT

**ACT robot learning with MediaPipe hand-tracking teleoperation.**

Control a simulated ALOHA robot arm using your webcam — no extra hardware needed. Collect demonstrations, train an ACT (Action Chunking with Transformers) policy via LeRobot, and evaluate it in the gym-aloha simulator.

```
webcam → MediaPipe Hands → 6-DOF pose delta → gym-aloha → LeRobot dataset → ACT training → policy
```

## Why this stack

| Component | Choice | Reason |
|---|---|---|
| Sim + task | `gym-aloha` (MuJoCo) | Free, realistic, standard ALOHA task suite |
| Data format | LeRobot dataset | Field-standard schema; one-click Hub publishing |
| Policy | ACT (CVAE + Transformer) | Fast to train, less data than Diffusion Policy |
| Teleop input | MediaPipe Hands (webcam) | Free, reuses computer-vision skills, better demo video than keyboard |

---

## Setup

```bash
# Python 3.10+
pip install -r requirements.txt
pip install -e .
```

A free Colab T4 or Kaggle P100 GPU is sufficient for training.

---

## 1 — Collect demonstrations

Put your hand in front of your webcam. The OpenCV preview window shows the skeleton overlay and gripper state.

| Gesture | Effect |
|---|---|
| Move hand left / right / forward / back | End-effector XYZ delta |
| Tilt / rotate wrist | Roll / pitch / yaw delta |
| Pinch thumb + index close | Close gripper |
| Spread thumb + index | Open gripper |

```bash
python scripts/collect_demos.py \
    --task gym_aloha/AlohaInsertion-v0 \
    --num-episodes 50 \
    --output-dir data/demos

# Optional: publish to the HuggingFace Hub
python scripts/collect_demos.py \
    --task gym_aloha/AlohaInsertion-v0 \
    --num-episodes 50 \
    --repo-id <your-hf-username>/handact-demos
```

---

## 2 — Train ACT

```bash
# Default config (configs/train.yaml)
python scripts/train.py

# Override on the command line (Hydra syntax)
python scripts/train.py training.batch_size=16 device=cpu
```

Checkpoints are saved to `outputs/train/` every 10 epochs. Training 100 epochs on 50 episodes takes ~1 hour on a T4.

Key hyperparameter to tune first: `policy.chunk_size` (default 100). Increase if the robot jerks between chunks; decrease if it's too slow to react.

---

## 3 — Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint outputs/train/policy_final.pt \
    --num-episodes 20 \
    --render
```

Reports per-episode success / reward and an overall success rate. The `--render` flag opens a live MuJoCo viewer.

---

## Project structure

```
HandACT/
├── handact/
│   └── teleop.py          # MediaPipe hand-tracking → 6-DOF action delta
├── scripts/
│   ├── collect_demos.py   # Teleoperated demo collection → LeRobot dataset
│   ├── train.py           # ACT training loop (Hydra config)
│   └── evaluate.py        # Chunk execution + temporal ensembling eval
├── configs/
│   ├── train.yaml         # Top-level training config
│   ├── env/
│   │   └── aloha_insertion.yaml
│   └── policy/
│       └── act.yaml       # ACT architecture + CVAE hyperparameters
└── requirements.txt
```

---

## How ACT works (quick reference)

1. **Vision encoder** (ResNet-18) → image feature embeddings
2. **CVAE encoder** (training only) → latent style variable *z* from demo action sequence
3. **Transformer decoder** takes `[image features, joint state, z]` → predicts a *chunk* of future actions (default: 100 steps ahead)
4. **Chunk execution**: the robot runs the predicted chunk, then re-queries the policy
5. **Temporal ensembling**: overlapping chunks are blended via exponential decay weights, smoothing transitions

---

## Upgrade paths

- **Better demos**: SpaceMouse (~$50) or Meta Quest controller for more precise control
- **More tasks**: swap `gym_aloha/AlohaInsertion-v0` for `AlohaTransferCube-v0` or any other LeRobot-compatible env
- **Stronger policy**: switch to Diffusion Policy if ACT averages between two strategies (bimodal demonstrations)
- **Real hardware**: LeRobot supports ALOHA hardware; the dataset format is identical
