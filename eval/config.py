from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Literal


LocalSearchMode = Literal["off", "auto", "force"]
LocalScoreMode = Literal["max", "top2_mean"]


@dataclass
class EvalConfig:
    # -------- Data / dataset --------
    csv_path: str | Path = Path("dataset/dataset.csv")
    source: str = "allen"  # "allen" | "real" | "both"
    limit: Optional[int] = None

    # -------- Volumes --------
    allen_cache_dir: str = "volume/data/allen"
    allen_res_um: int = 25
    real_volume_path: str | Path | None = Path("volume/data/real/registered_brain_25um.nii.gz")

    # -------- Slice sampling (eval-time) --------
    size_px: int = 512
    pixel_step_vox: float = 1.0
    linear_interp: bool = True

    # -------- Search --------
    angles: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    final_k: int = 100
    k_per_angle: int = 64
    crop_foreground: bool = True
    debug: bool = False

    flip_x: bool = True
    flip_y: bool = True
    pad_to_square: bool = True

    # High-level evaluation label, useful when running multiple modes
    search_mode_label: str = "fast"

    # New query expansion controls
    local_search_mode: LocalSearchMode = "off"
    local_k_per_view: Optional[int] = None
    local_score_mode: LocalScoreMode = "top2_mean"
    local_weight: float = 0.35
    global_weight: float = 1.0

    auto_local_aspect_threshold: float = 1.35
    local_crop_overlap: float = 0.50
    local_crop_min_side_px: int = 64
    auto_max_local_crops: int = 3
    force_max_local_crops: int = 8
    force_square_scales: Tuple[int, ...] = (2,)

    # -------- Geometry distance (delegated to volume_helper.Slice.distance) --------
    distance_grid: int = 64
    distance_trim_frac: float = 0.10
    distance_physical: bool = False  # must be False (voxel units) to avoid spacing mismatches

    # -------- Corner Chamfer (3 points) in microns --------
    # If None => use allen_res_um (e.g., 25um/voxel)
    corner_um_per_vox: Optional[float] = None

    # -------- Adaptive temperature for listwise targets --------
    tau_q_lo: float = 0.10
    tau_q_hi: float = 0.90
    tau_div: float = 3.0
    tau_min: float = 1e-3
    tau_max: float = 1e4

    # -------- Optional reranker passthrough --------
    use_reranker: bool = False
    rerank_topk: Optional[int] = None
    reranker_model_path: Path = Path("out/reranker/reranker.pt")
    reranker_device: str = "cuda"
    reranker_batch_size: int = 32

    # -------- Output --------
    save_dir: Path = Path("out/eval")
    save_k: Optional[int] = None
    save_seed: int = 42

    # -------- Resume / overwrite --------
    overwrite: bool = False

    # -------- Memory / streaming safety --------
    max_retrieved_slice_cache: int = 256
    csv_flush_every: int = 128
    gc_every_rows: int = 25
