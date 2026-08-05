"""Training / evaluation loop for MaSIF-site."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn

from masif.config import ModelConfig, TrainConfig
from masif.models.masif_site import MaSIFSite
from masif.patches import ProteinPatches, build_batch

logger = logging.getLogger(__name__)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _eval_full(
    model: MaSIFSite,
    patches: ProteinPatches,
    device: torch.device,
    feat_mask,
) -> np.ndarray:
    """Interface scores for every vertex of a protein."""
    import torch as t

    x = patches.input_feat
    if int(np.asarray(feat_mask).sum()) < 5:
        keep = np.where(np.asarray(feat_mask) > 0)[0]
        x = x[:, :, keep]
    with t.no_grad():
        model.eval()
        x = t.from_numpy(x).to(device)
        rho = t.from_numpy(patches.rho).to(device)
        theta = t.from_numpy(patches.theta).to(device)
        mask = t.from_numpy(patches.mask).to(device)
        indices = t.from_numpy(patches.indices).to(device)
        scores = model.score(x, rho, theta, mask, indices).cpu().numpy()
    return scores


def train(
    model: MaSIFSite,
    files: List[Path],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    out_dir: Path,
) -> dict:
    """Train ``model`` over precomputed proteins for ``train_cfg.epochs`` epochs.

    * One optimizer step is taken per protein on a balanced subset of its interface /
      non-interface vertices (mirroring the original per-protein batching).
    * ``val_fraction`` of proteins are held out; the checkpoint with the best mean
      validation AUC is saved together with a JSON history.
    """
    rng = random.Random(train_cfg.seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * train_cfg.val_fraction))
    val_files = files[:n_val]
    train_files = files[n_val:]

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_cfg.learning_rate, weight_decay=model_cfg.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    history = []

    for epoch in range(train_cfg.epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0
        for f in train_files:
            patches = ProteinPatches.load(f)
            if len(patches.labels) > train_cfg.max_vertices:
                continue
            if (
                patches.labels.sum() > train_cfg.max_pos_frac * len(patches.labels)
                or patches.labels.sum() < train_cfg.min_pos_vertices
            ):
                continue
            batch = build_batch(patches, train_cfg.batch_size, seed=epoch * 1000 + n_steps)
            x = batch["input_feat"].to(device)
            rho = batch["rho"].to(device)
            theta = batch["theta"].to(device)
            mask = batch["mask"].to(device)
            indices = batch["indices"].to(device)
            labels = batch["labels"].to(device)

            logits = model(x, rho, theta, mask, indices)
            loss = loss_fn(logits[:, 0], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1

        # validation AUC
        val_aucs = []
        all_labels, all_scores = [], []
        for f in val_files:
            patches = ProteinPatches.load(f)
            scores = _eval_full(model, patches, device, model_cfg.feat_mask)
            all_labels.append(patches.labels)
            all_scores.append(scores)
            val_aucs.append(_auc(patches.labels, scores))
        pooled_auc = _auc(
            np.concatenate(all_labels), np.concatenate(all_scores)
        )
        mean_val = float(np.nanmean(val_aucs)) if val_aucs else float("nan")

        logger.info(
            "epoch %d/%d loss=%.4f mean_val_auc=%.4f pooled_val_auc=%.4f",
            epoch + 1, train_cfg.epochs, running_loss / max(n_steps, 1), mean_val, pooled_auc,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "loss": running_loss / max(n_steps, 1),
                "mean_val_auc": mean_val,
                "pooled_val_auc": pooled_auc,
            }
        )

        if mean_val > best_val_auc:
            best_val_auc = mean_val
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "epoch": epoch + 1, "best_val_auc": best_val_auc},
                out_dir / "model.pt",
            )
            logger.info("saved best model (val_auc=%.4f)", best_val_auc)

    with (out_dir / "history.json").open("w") as fh:
        json.dump(history, fh, indent=2)
    return {"best_val_auc": best_val_auc}
