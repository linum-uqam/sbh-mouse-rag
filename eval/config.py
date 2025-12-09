# eval/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class EvalConfig:
    """
    Configuration for evaluation of the FAISS+DINOv3 index.

    This mirrors what used to be the __init__ signature of Evaluator,
    but keeps all the knobs in a single structured object with sane
    defaults.
    """

    # -------- Data / dataset --------
    csv_path: str | Path = Path("dataset/dataset.csv")
    source: str = "allen"                   # "allen" | "real" | "both"
    limit: Optional[int] = None
    include_annotation: bool = True

    # -------- Volumes --------
    allen_cache_dir: str = "volume/data/allen"
    allen_res_um: int = 25
    real_volume_path: str | Path | None = Path(
        "volume/data/real/registered_brain_25um.nii.gz"
    )

    # -------- Slice sampling (eval-time) --------
    size_px: int = 512
    pixel_step_vox: float = 1.0
    linear_interp: bool = True  # True = better, False = faster

    # -------- Search --------
    angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    final_k: int = 10
    k_per_angle: int = 64
    crop_foreground: bool = True
    debug: bool = False

    # -------- Reranker (optional) --------
    use_reranker: bool = False
    rerank_topk: Optional[int] = None
    reranker_model_path: Path = Path("out/reranker/reranker.pt")
    reranker_device: str = "cuda"
    reranker_batch_size: int = 32

    # -------- Output --------
    save_dir: Path = Path("out/eval")
    save_k: Optional[int] = None
    save_seed: int = 42

    @property
    def csv_path_str(self) -> str:
        return str(self.csv_path)

    @property
    def real_volume_path_str(self) -> Optional[str]:
        return None if self.real_volume_path is None else str(self.real_volume_path)
