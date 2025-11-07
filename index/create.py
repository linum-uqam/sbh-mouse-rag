# index/create.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict
import multiprocessing as mp

import faiss
import numpy as np
from tqdm.auto import tqdm

from volume.volume_helper import AllenVolume
from index.geom import iter_slices_fibonacci, count_slices_fibonacci
from index.utils import make_slice_id, image_to_token_mask
from index.store import build_flat_ip, build_hnsw_ip, write_manifest_slice, wrap_with_ids
from index.model.dino import model
from .config import OUT_DIR, TOK_DIR, K_NORMALS, SLICE_SIZE, USE_HNSW, D, HNSW_M, HNSW_EF_CONSTRUCTION

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)

    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)

    total_slices = count_slices_fibonacci(allen, k_normals=K_NORMALS)

    # Build FAISS coarse (global) index only
    coarse = wrap_with_ids(build_hnsw_ip(D) if USE_HNSW else build_flat_ip(D))

    slice_rows: List[Dict] = []

    it = iter_slices_fibonacci(
        allen,
        k_normals=K_NORMALS,
        size_px=SLICE_SIZE,
        linear_interp=True,
        include_annotation=False,
    )

    bar = tqdm(it, total=total_slices, desc="Indexing slices", unit="slice", dynamic_ncols=True)
    
    for slc, meta in bar:
        bar.set_postfix(normal=meta["normal_idx"], depth=meta["depth_idx"])
        sid = make_slice_id(meta["normal_idx"], meta["depth_idx"], meta["rot_idx"])
        if not allen.is_valid_slice(slc):
            continue

        slc_n = slc.normalized()

        # Embeddings (global + tokens)
        out = model.embed_both(slc_n.image)
        g = out["global"].astype(np.float32)
        t = out["tokens"].astype(np.float32)      # (T,D)
        grid_hw = int(np.sqrt(t.shape[0]))

        # --- NEW: candidate-side patch mask (True=foreground) ---
        fg_hw = image_to_token_mask(slc_n.image, grid_hw=grid_hw, bg_threshold=0.02)  # tune 0.01-0.05 if needed
        mask_flat = fg_hw.reshape(-1).astype(np.uint8)  # compact

        # Normalize global and add to coarse index
        faiss.normalize_L2(g.reshape(1, -1))
        coarse.add_with_ids(g.reshape(1, -1), np.array([sid], dtype=np.int64))

        # Persist token matrix (float16) + mask (uint8)
        tok_path  = TOK_DIR / f"n{meta['normal_idx']}_d{meta['depth_idx']}_r{meta['rot_idx']}.fp16.npy"
        mask_path = TOK_DIR / f"n{meta['normal_idx']}_d{meta['depth_idx']}_r{meta['rot_idx']}.mask.npy"
        np.save(tok_path,  t.astype(np.float16))
        np.save(mask_path, mask_flat)

        # Slice manifest row
        slice_rows.append({
            "id": sid,
            "token_path": str(tok_path),
            "mask_path":  str(mask_path),     # <-- NEW

            # pose
            "normal_idx": meta["normal_idx"], "depth_idx": meta["depth_idx"], "rot_idx": meta["rot_idx"],
            "normal_x": meta["normal_xyz_unit"][0], "normal_y": meta["normal_xyz_unit"][1], "normal_z": meta["normal_xyz_unit"][2],
            "depth_vox": meta["depth_vox"], "rotation_deg": meta["rotation_deg"],

            # token grid
            "grid_h": grid_hw, "grid_w": grid_hw, "feat_dim": t.shape[1],

            "size_px": int(SLICE_SIZE),
            "resolution_um": 25,
            "linear_interp": True,
        })

        # Persist incremental artifacts for crash-safety
        faiss.write_index(coarse, str(OUT_DIR / "coarse.faiss"))
        write_manifest_slice(slice_rows, OUT_DIR / "manifest.parquet")

    print("Done.")
    print(f" - Slices:  {len(slice_rows)}")
    print(f" - Saved:   {OUT_DIR/'coarse.faiss'}")
    print(f" - Manifest:{OUT_DIR/'manifest.parquet'}")


if __name__ == "__main__":
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    main()
