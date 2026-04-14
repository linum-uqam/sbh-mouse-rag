from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig, RerankerConfig
from .data import prepare_dataloaders
from .model import ListwiseReranker


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reranker] Seed set to {seed}")


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    very_neg = torch.finfo(logits.dtype).min / 2
    x = logits.masked_fill(~mask, very_neg)
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    exp_x = torch.exp(x).masked_fill(~mask, 0.0)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return x - torch.log(denom)


def masked_distance_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    mask_f = mask.to(dtype=pred.dtype)
    if loss_type == "mse":
        per = (pred - target) ** 2
    else:
        per = torch.nn.functional.huber_loss(pred, target, reduction="none")
    per = per * mask_f
    denom = mask_f.sum().clamp_min(1.0)
    return per.sum() / denom


class TrainingRun:
    def __init__(self, cfg: TrainingConfig, model_cfg: RerankerConfig | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg or RerankerConfig(device=cfg.device)
        self.device = torch.device(cfg.device)
        self.model: ListwiseReranker | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def run(self) -> Dict[str, float]:
        self.cfg.validate()
        set_global_seed(self.cfg.seed)
        self.train_loader, self.val_loader, self.test_loader = prepare_dataloaders(
            self.cfg, embed_dim=int(self.model_cfg.embed_dim)
        )
        self._setup_model()

        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.cfg.num_epochs + 1):
            tr = self._run_epoch(self.train_loader, train=True, epoch=epoch, phase="train")
            va = self._run_epoch(self.val_loader, train=False, epoch=epoch, phase="val")

            current_lr = self._get_current_lr()
            row = {
                "epoch": float(epoch),
                "lr": float(current_lr),
                "train_loss": float(tr["loss"]),
                "train_rank_loss": float(tr["rank_loss"]),
                "train_dist_loss": float(tr["dist_loss"]),
                "val_loss": float(va["loss"]),
                "val_rank_loss": float(va["rank_loss"]),
                "val_dist_loss": float(va["dist_loss"]),
            }
            history.append(row)

            print(
                f"[Reranker] epoch={epoch:03d} | lr={current_lr:.6g} | "
                f"train_loss={tr['loss']:.6f} (rank={tr['rank_loss']:.6f}, dist={tr['dist_loss']:.6f}) | "
                f"val_loss={va['loss']:.6f} (rank={va['rank_loss']:.6f}, dist={va['dist_loss']:.6f})"
            )

            if self.scheduler is not None:
                self.scheduler.step(float(va["loss"]))

            improved = float(va["loss"]) < (best_val_loss - float(self.cfg.early_stopping_min_delta))
            if improved:
                best_val_loss = float(va["loss"])
                best_epoch = epoch
                epochs_without_improvement = 0
                assert self.model is not None
                self.model.save(self.cfg.out_path)
            else:
                epochs_without_improvement += 1
                print(
                    f"[Reranker] no significant val improvement for {epochs_without_improvement} epoch(s) "
                    f"(best={best_val_loss:.6f} @ epoch {best_epoch}, min_delta={self.cfg.early_stopping_min_delta})"
                )

            if epochs_without_improvement >= int(self.cfg.early_stopping_patience):
                print(
                    f"[Reranker] Early stopping triggered at epoch {epoch:03d} "
                    f"after {epochs_without_improvement} epoch(s) without significant validation improvement."
                )
                break

        self.model = ListwiseReranker.load(self.cfg.out_path, map_location=self.device)
        te = self._run_epoch(self.test_loader, train=False, epoch=len(history) + 1, phase="test")
        print(
            f"[Reranker] final_test_loss={te['loss']:.6f} "
            f"(rank={te['rank_loss']:.6f}, dist={te['dist_loss']:.6f})"
        )
        self._save_history_csv(history, best_val_loss, te["loss"], best_epoch)
        return {"best_val_loss": best_val_loss, "test_loss": te["loss"], "best_epoch": float(best_epoch)}

    def _setup_model(self) -> None:
        self.model = ListwiseReranker(cfg=self.model_cfg)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
        if bool(self.cfg.use_plateau_scheduler):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=float(self.cfg.plateau_factor),
                patience=int(self.cfg.plateau_patience),
                min_lr=float(self.cfg.plateau_min_lr),
            )
        else:
            self.scheduler = None

        print(
            "[Reranker] Model initialized:\n"
            f"  - embed_dim             : {self.model_cfg.embed_dim}\n"
            f"  - hidden_dims           : {self.model_cfg.hidden_dims}\n"
            f"  - dropout               : {self.model_cfg.dropout}\n"
            f"  - pair_features         : {self.model_cfg.use_pair_features}\n"
            f"  - list_k                : {self.cfg.list_k}\n"
            f"  - train_topk            : {self.cfg.train_topk}\n"
            f"  - sampling_mode         : {self.cfg.sampling_mode}\n"
            f"  - distance_loss_weight  : {self.cfg.distance_loss_weight}\n"
            f"  - distance_loss_type    : {self.cfg.distance_loss_type}\n"
            f"  - distance_target       : {self.cfg.distance_target}\n"
            f"  - lr                    : {self.cfg.lr}\n"
            f"  - weight_decay          : {self.cfg.weight_decay}\n"
            f"  - grad_clip_norm        : {self.cfg.grad_clip_norm}\n"
            f"  - amp                   : {self.use_amp}\n"
            f"  - early_stop_patience   : {self.cfg.early_stopping_patience}\n"
            f"  - early_stop_min_delta  : {self.cfg.early_stopping_min_delta}\n"
            f"  - plateau_scheduler     : {self.cfg.use_plateau_scheduler}\n"
            f"  - plateau_patience      : {self.cfg.plateau_patience}\n"
            f"  - plateau_factor        : {self.cfg.plateau_factor}\n"
            f"  - plateau_min_lr        : {self.cfg.plateau_min_lr}"
        )

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int, phase: str) -> Dict[str, float]:
        assert self.model is not None
        assert self.optimizer is not None
        assert loader is not None
        self.model.train(train)

        total_loss = 0.0
        total_rank = 0.0
        total_dist = 0.0
        total_count = 0
        iterator = tqdm(loader, desc=f"Reranker(phase-a) | epoch={epoch:03d} | phase={phase}")

        for batch in iterator:
            q_emb = batch["q_emb"].to(self.device, non_blocking=True)
            c_emb = batch["c_emb"].to(self.device, non_blocking=True)
            p_gt = batch["gt_prob"].to(self.device, non_blocking=True)
            d_gt = batch["gt_dist"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            p_gt = torch.where(mask, p_gt, torch.zeros_like(p_gt))
            denom = p_gt.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p_gt = p_gt / denom

            autocast_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
            with torch.amp.autocast(device_type=autocast_device, enabled=self.use_amp):
                rank_logits, dist_pred = self.model(q_emb, c_emb)
                log_q = masked_log_softmax(rank_logits, mask, dim=1)
                rank_loss = -(p_gt * log_q).sum(dim=1).mean()
                if bool(self.cfg.use_distance_loss):
                    dist_loss = masked_distance_loss(dist_pred, d_gt, mask, self.cfg.distance_loss_type)
                    loss = rank_loss + float(self.cfg.distance_loss_weight) * dist_loss
                else:
                    dist_loss = torch.zeros((), device=self.device, dtype=rank_loss.dtype)
                    loss = rank_loss

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
                    self.optimizer.step()

            bs = int(q_emb.shape[0])
            total_loss += float(loss.item()) * bs
            total_rank += float(rank_loss.item()) * bs
            total_dist += float(dist_loss.item()) * bs
            total_count += bs
            iterator.set_postfix_str(
                f"loss={total_loss/max(total_count,1):.6f} | "
                f"rank={total_rank/max(total_count,1):.6f} | "
                f"dist={total_dist/max(total_count,1):.6f}"
            )

        denom = max(total_count, 1)
        return {
            "loss": total_loss / denom,
            "rank_loss": total_rank / denom,
            "dist_loss": total_dist / denom,
        }

    def _get_current_lr(self) -> float:
        assert self.optimizer is not None
        lrs = [float(pg.get("lr", 0.0)) for pg in self.optimizer.param_groups]
        return lrs[0] if lrs else 0.0

    def _save_history_csv(self, history: List[Dict[str, float]], best_val: float, test_loss: float, best_epoch: int) -> None:
        out_path = Path(self.cfg.out_path)
        hist_path = out_path.with_suffix(".history.csv")
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with hist_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "lr",
                    "train_loss",
                    "train_rank_loss",
                    "train_dist_loss",
                    "val_loss",
                    "val_rank_loss",
                    "val_dist_loss",
                ],
            )
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print(
            "[Reranker] Saved loss history:\n"
            f"  - path            : {hist_path}\n"
            f"  - best_epoch      : {best_epoch}\n"
            f"  - best_val_loss   : {best_val:.6f}\n"
            f"  - final_test_loss : {test_loss:.6f}"
        )
