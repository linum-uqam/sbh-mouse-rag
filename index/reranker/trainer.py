from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
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
    """
    logits: (..., K)
    mask:   (..., K) bool
    returns log_softmax over valid positions; invalid positions are -inf.
    """
    mask = mask.to(dtype=torch.bool)
    very_neg = torch.finfo(logits.dtype).min / 2
    x = logits.masked_fill(~mask, very_neg)

    # stable log-softmax (only valid entries contribute)
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    exp_x = torch.exp(x).masked_fill(~mask, 0.0)
    denom = exp_x.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return x - torch.log(denom)


class TrainingRun:
    def __init__(self, cfg: TrainingConfig, model_cfg: RerankerConfig | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg or RerankerConfig(device=cfg.device)

        self.device = torch.device(cfg.device)
        self.model: ListwiseReranker | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def run(self) -> Dict[str, float]:
        set_global_seed(self.cfg.seed)

        # Data
        self.train_loader, self.val_loader, self.test_loader = prepare_dataloaders(
            self.cfg, embed_dim=int(self.model_cfg.embed_dim)
        )

        # Model
        self._setup_model()

        best_val_loss = float("inf")
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self._run_epoch(self.train_loader, train=True, epoch=epoch, phase="train")
            val_loss = self._run_epoch(self.val_loader, train=False, epoch=epoch, phase="val")

            history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})

            print(f"[Reranker] epoch={epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                assert self.model is not None
                self.model.save(self.cfg.out_path)

        # Test (load best)
        self.model = ListwiseReranker.load(self.cfg.out_path, map_location=self.device)
        test_loss = self._run_epoch(self.test_loader, train=False, epoch=self.cfg.num_epochs + 1, phase="test")
        print(f"[Reranker] final_test_loss={test_loss:.6f}")

        self._save_history_csv(history, best_val_loss, test_loss)
        return {"best_val_loss": best_val_loss, "test_loss": test_loss}

    def _setup_model(self) -> None:
        self.model = ListwiseReranker(cfg=self.model_cfg)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

        print(
            "[Reranker] Model initialized:\n"
            f"  - embed_dim      : {self.model_cfg.embed_dim}\n"
            f"  - hidden_dims    : {self.model_cfg.hidden_dims}\n"
            f"  - dropout        : {self.model_cfg.dropout}\n"
            f"  - pair_features  : {self.model_cfg.use_pair_features}\n"
            f"  - list_k         : {self.cfg.list_k}\n"
            f"  - train_topk     : {self.cfg.train_topk}\n"
            f"  - lr             : {self.cfg.lr}\n"
            f"  - weight_decay   : {self.cfg.weight_decay}\n"
            f"  - grad_clip_norm : {self.cfg.grad_clip_norm}\n"
            f"  - amp            : {self.use_amp}"
        )

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int, phase: str) -> float:
        assert self.model is not None
        assert self.optimizer is not None
        assert loader is not None

        self.model.train(train)

        total_loss = 0.0
        total_count = 0

        iterator = tqdm(loader, desc=f"Reranker(listwise) | epoch={epoch:03d} | phase={phase}")

        for batch in iterator:
            q_emb = batch["q_emb"].to(self.device, non_blocking=True)      # (B,D)
            c_emb = batch["c_emb"].to(self.device, non_blocking=True)      # (B,K,D)
            p_gt  = batch["gt_prob"].to(self.device, non_blocking=True)    # (B,K)
            mask  = batch["mask"].to(self.device, non_blocking=True)       # (B,K)

            # ensure p_gt is valid / normalized over mask (defensive)
            p_gt = torch.where(mask, p_gt, torch.zeros_like(p_gt))
            denom = p_gt.sum(dim=1, keepdim=True).clamp_min(1e-12)
            p_gt = p_gt / denom

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(q_emb, c_emb)  # (B,K)
                log_q = masked_log_softmax(logits, mask, dim=1)  # (B,K)
                loss = -(p_gt * log_q).sum(dim=1).mean()

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
            total_count += bs

            iterator.set_postfix_str(f"loss={total_loss / max(total_count,1):.6f}")

        return total_loss / max(total_count, 1)

    def _save_history_csv(self, history: List[Dict[str, float]], best_val: float, test_loss: float) -> None:
        out_path = Path(self.cfg.out_path)
        hist_path = out_path.with_suffix(".history.csv")
        hist_path.parent.mkdir(parents=True, exist_ok=True)

        with hist_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        print(
            "[Reranker] Saved loss history:\n"
            f"  - path            : {hist_path}\n"
            f"  - best_val_loss   : {best_val:.6f}\n"
            f"  - final_test_loss : {test_loss:.6f}"
        )
