# eval/run.py
from __future__ import annotations
import argparse
import multiprocessing as mp
import faiss

from eval.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset-driven eval loop over FAISS+DINOv3 index")
    # Data
    p.add_argument("--csv", type=str, default="dataset/dataset.csv")
    p.add_argument("--source", type=str, default="allen", choices=["allen", "real", "both"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include-annotation", action="store_true", default=True)

    # Volumes
    p.add_argument("--allen-cache-dir", type=str, default="volume/data/allen")
    p.add_argument("--allen-res-um", type=int, default=25)
    p.add_argument("--real-volume-path", type=str, default="volume/data/real/real_mouse_brain_ras_25um.nii.gz")

    # Slice sampling
    p.add_argument("--size-px", type=int, default=512)
    p.add_argument("--pixel-step-vox", type=float, default=1.0)
    p.add_argument("--linear-interp", action="store_true", default=True)

    # Search/index
    p.add_argument("--mode", type=str, default="col", choices=["base", "col"])
    p.add_argument("--angles", type=float, nargs="+", default=[0, 90, 180, 270])
    p.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 14])
    p.add_argument("--final-k", type=int, default=10)
    p.add_argument("--base-per-rotation", type=int, default=200)
    p.add_argument("--base-topk-for-col", type=int, default=300)
    p.add_argument("--use-token-ann", action="store_true", default=True)
    p.add_argument("--token-scales", type=int, nargs="+", default=[4, 8, 14])
    p.add_argument("--token-topM", type=int, default=32)
    p.add_argument("--debug", action="store_true", default=False)

    # Output
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--save-k", type=int, default=None,
                   help="Save exactly K random rows among the processed set. If omitted, save all processed rows.")
    p.add_argument("--save-seed", type=int, default=123)

    return p.parse_args()


def main():
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    a = parse_args()

    Evaluator(
        csv_path=a.csv,
        source=a.source,
        limit=a.limit,
        include_annotation=a.include_annotation,
        allen_cache_dir=a.allen_cache_dir,
        allen_res_um=a.allen_res_um,
        real_volume_path=a.real_volume_path,
        size_px=a.size_px,
        pixel_step_vox=a.pixel_step_vox,
        linear_interp=a.linear_interp,
        mode=a.mode,
        angles=tuple(a.angles),
        scales=tuple(a.scales),
        final_k=a.final_k,
        base_per_rotation=a.base_per_rotation,
        base_topk_for_col=a.base_topk_for_col,
        use_token_ann=a.use_token_ann,
        token_scales=tuple(a.token_scales),
        token_topM=a.token_topM,
        debug=a.debug,
        save_dir=a.save_dir,
        save_k=a.save_k,
        save_seed=a.save_seed,
    ).run()


if __name__ == "__main__":
    main()
