# index/reranker/trainer.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
import csv

from .model import TwoTowerReranker, RerankerConfig
from .dataset import ImagePairDataset, VolumePairDataset
from index.utils import log
from volume.volume_helper import AllenVolume, NiftiVolume


@dataclass
class TrainingConfig:
    """
    Configuration for a reranker training run.
    """

    # --- data selection ---
    data_mode: str = "auto"        # "auto" | "csv" | "volume"
    csv_path: Optional[str] = None

    # Volume-based mode options
    allen_cache_dir: str = "volume/data/allen"
    allen_resolution_um: int = 25
    real_nifti_path: Optional[str] = None
    n_samples: int = 50000
    slice_size: int = 224

    # --- training ---
    out_path: str = "out/reranker/reranker.pt"
    batch_size: int = 32
    num_epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda"
    num_workers: int = 4  # only used for CSV mode; volume mode uses 0
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingRun:
    """
    Encapsulates a full reranker training run:
      - dataset loading + splitting
      - model setup
      - training loop (MSE regression)
      - final evaluation + saving
    """

    def __init__(self, cfg: TrainingConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

        self.model: TwoTowerReranker | None = None
        self.criterion: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    # ---------- top-level entry point ----------

    def run(self) -> Dict[str, float]:
        """
        Run the full training loop and return final metrics.
        """
        self._set_seed(self.cfg.seed)
        self._setup_datasets()
        self._setup_model()

        best_val_loss = float("inf")
        history: list[dict[str, float]] = []

        for epoch in range(1, self.cfg.num_epochs + 1):
            train_loss = self._run_epoch(
                self.train_loader, train=True, epoch=epoch, phase="train"
            )
            val_loss = self._run_epoch(
                self.val_loader, train=False, epoch=epoch, phase="val"
            )

            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                }
            )

            log(
                f"Reranker | epoch={epoch:03d} "
                f"| train_loss={train_loss:.6f} "
                f"| val_loss={val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save(self.cfg.out_path)

        # Reload best model for test evaluation
        self.model = TwoTowerReranker.load(
            self.cfg.out_path,
            encoder=None,  # default encoder (dino_model)
            map_location=self.device,
        )
        test_loss = self._run_epoch(
            self.test_loader, train=False, epoch=self.cfg.num_epochs + 1, phase="test"
        )
        log(f"Reranker | final_test_loss={test_loss:.6f}")

        # Save history as CSV next to model
        self._save_history_csv(history, best_val_loss, test_loss)

        return {
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
        }



    # ---------- setup helpers ----------

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log(f"[Reranker] Seed set to {seed}")

    def _setup_datasets(self) -> None:
        mode = self.cfg.data_mode.lower()
        if mode == "csv":
            self._setup_datasets_from_csv()
        elif mode == "volume":
            self._setup_datasets_from_volume()
        elif mode == "auto":
            if self.cfg.csv_path is not None and Path(self.cfg.csv_path).exists():
                log("[Reranker] data_mode=auto -> using CSV because file exists.")
                self._setup_datasets_from_csv()
            else:
                log("[Reranker] data_mode=auto -> CSV missing, falling back to volume mode.")
                self._setup_datasets_from_volume()
        else:
            raise ValueError(f"Unknown data_mode={self.cfg.data_mode!r}; expected 'auto', 'csv' or 'volume'.")

    def _setup_datasets_from_csv(self) -> None:
        assert self.cfg.csv_path is not None, "csv_path must be set in CSV mode."
        csv_path = Path(self.cfg.csv_path)
        log(f"[Reranker] Loading dataset from CSV: {csv_path}")
        full_ds = ImagePairDataset(csv_path)

        n = len(full_ds)
        n_train = int(round(self.cfg.train_frac * n))
        n_val = int(round(self.cfg.val_frac * n))
        n_test = n - n_train - n_val
        log(f"[Reranker] Dataset sizes (CSV): total={n}, train={n_train}, val={n_val}, test={n_test}")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_ds, [n_train, n_val, n_test]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def _setup_datasets_from_volume(self) -> None:
        # Ensure we have a real volume path
        if self.cfg.real_nifti_path is None:
            raise ValueError(
                "real_nifti_path must be provided for volume mode "
                "(e.g. 'volume/data/real/registered_brain_25um.nii.gz')."
            )

        log("[Reranker] Initializing AllenVolume and NiftiVolume for volume-based dataset...")
        allen = AllenVolume(
            cache_dir=self.cfg.allen_cache_dir,
            resolution_um=self.cfg.allen_resolution_um,
        )
        real = NiftiVolume(self.cfg.real_nifti_path)

        full_ds = VolumePairDataset(
            allen=allen,
            real=real,
            n_samples=self.cfg.n_samples,
            slice_size=self.cfg.slice_size,
            # multi-scale + augmentation enabled by default
            enable_crops=True,
            crop_prob=0.8,
            enable_augment=True,
        )

        n = len(full_ds)
        n_train = int(round(self.cfg.train_frac * n))
        n_val = int(round(self.cfg.val_frac * n))
        n_test = n - n_train - n_val
        log(f"[Reranker] Dataset sizes (volume): total={n}, train={n_train}, val={n_val}, test={n_test}")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_ds, [n_train, n_val, n_test]
        )

        # num_workers=0 to avoid copying big volumes around
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def _setup_model(self) -> None:
        reranker_cfg = RerankerConfig(device=self.cfg.device)
        self.model = TwoTowerReranker(cfg=reranker_cfg)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        # Only train the head (encoder is frozen)
        self.optimizer = torch.optim.Adam(self.model.head.parameters(), lr=self.cfg.lr)

        log(
            "[Reranker] Model initialized: "
            f"embed_dim={reranker_cfg.embed_dim}, hidden={tuple(reranker_cfg.hidden_dims)}, "
            f"device={self.cfg.device}, lr={self.cfg.lr}"
        )

    def _save_history_csv(
        self,
        history: list[dict[str, float]],
        best_val: float,
        test_loss: float,
    ) -> None:
        """
        Save per-epoch train/val loss progression to a simple CSV file.
        """
        out_path = Path(self.cfg.out_path)
        hist_path = out_path.with_suffix(".history.csv")
        hist_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["epoch", "train_loss", "val_loss"]
        with hist_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        log(f"Reranker | saved loss history to {hist_path}")


    # ---------- epoch loop ----------

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        epoch: int,
        phase: str,
    ) -> float:
        assert self.model is not None
        assert self.criterion is not None
        assert self.optimizer is not None

        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_count = 0

        iterator = tqdm(
            loader,
            desc=f"Reranker | epoch={epoch:03d} | phase={phase}",
            ncols=100,
        )

        for batch in iterator:
            q_imgs = batch["q_img"]
            c_imgs = batch["c_img"]
            targets = torch.as_tensor(batch["target"], dtype=torch.float32, device=self.device)

            with torch.no_grad():
                q_emb = self.model.encode_batch(q_imgs)
                c_emb = self.model.encode_batch(c_imgs)

            preds = self.model(q_emb, c_emb)
            loss = self.criterion(preds, targets)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            bs = targets.shape[0]
            total_loss += float(loss.item()) * bs
            total_count += bs

            iterator.set_postfix(loss=float(loss.item()))

        return total_loss / max(total_count, 1)


