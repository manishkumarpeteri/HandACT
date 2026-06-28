"""
Train an ACT policy on collected demonstrations.

Usage:
    python scripts/train.py                            # uses configs/train.yaml defaults
    python scripts/train.py training.batch_size=16     # Hydra override
    python scripts/train.py device=cpu                 # CPU-only (Colab / Kaggle)
"""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    log.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------
    # Imports deferred so Hydra can override before anything initialises
    # ------------------------------------------------------------------
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from lerobot.common.policies.act.configuration_act import ACTConfig
    from torch.utils.data import DataLoader

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu"
                          else "cpu")
    log.info(f"Training on {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = LeRobotDataset(
        repo_id=cfg.dataset_repo_id,
        root=cfg.root,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    log.info(f"Dataset: {len(dataset)} frames across {dataset.num_episodes} episodes")

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    policy_cfg = ACTConfig(**OmegaConf.to_container(cfg.policy, resolve=True))
    policy = ACTPolicy(policy_cfg, dataset_stats=dataset.stats)
    policy.to(device)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.num_epochs
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.training.num_epochs + 1):
        policy.train()
        epoch_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = policy.forward(batch)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg.training.grad_clip_norm
            )
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch}/{cfg.training.num_epochs}  loss={avg_loss:.4f}")

        if epoch % cfg.save_freq == 0:
            ckpt = log_dir / f"checkpoint_epoch{epoch:04d}.pt"
            torch.save(policy.state_dict(), ckpt)
            log.info(f"  Saved checkpoint: {ckpt}")

    # Final save
    final_ckpt = log_dir / "policy_final.pt"
    torch.save(policy.state_dict(), final_ckpt)
    log.info(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
