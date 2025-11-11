from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from tqdm.auto import tqdm

from volume.volume_helper import AllenVolume
from index.geom import iter_slices_fibonacci, count_slices_fibonacci
from index.utils import make_slice_id, image_to_token_mask, log
from index.store import write_manifest_slice, IndexStore
from index.builder import (
    build_base_index, adaptive_pool_tokens_to_vec,
    create_ivfpq,
)
from index.model.dino import model
from .config import (
    OUT_DIR, TOK_DIR, K_NORMALS, SLICE_SIZE, D,
    USE_HNSW, HNSW_M, HNSW_EF_CONSTRUCTION,
    SLICE_SCALES,
    TOKEN_INDEX_PATH,
)



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOK_DIR.mkdir(parents=True, exist_ok=True)

    allen = AllenVolume(cache_dir="volume/data/allen", resolution_um=25)
    total_slices = count_slices_fibonacci(allen, k_normals=K_NORMALS)

    # ---- Global (1x1) index and mandatory per-scale indexes ----
    coarse = build_base_index(D, USE_HNSW, HNSW_M, HNSW_EF_CONSTRUCTION)
    coarse_by_scale: Dict[int, faiss.Index] = {k: build_base_index(D, USE_HNSW, HNSW_M, HNSW_EF_CONSTRUCTION)
                                               for k in SLICE_SCALES}

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
        g = out["global"].astype(np.float32)     # (D,)
        t = out["tokens"].astype(np.float32)     # (T,D)
        grid_hw = int(np.sqrt(t.shape[0]))

        # Candidate-side patch mask (True=foreground)
        fg_hw = image_to_token_mask(slc_n.image, grid_hw=grid_hw, bg_threshold=0.02)
        mask_flat = fg_hw.reshape(-1).astype(np.uint8)

        # ---- add to global index ----
        g1 = g.reshape(1, -1)
        faiss.normalize_L2(g1)
        coarse.add_with_ids(g1, np.array([sid], dtype=np.int64))

        # ---- add to per-scale indexes (pooled vectors) ----
        # pool tokens -> one (D,) vector per k, normalize, add with same sid
        for k in SLICE_SCALES:
            vk = adaptive_pool_tokens_to_vec(t, grid_hw, k).reshape(1, -1)
            faiss.normalize_L2(vk)
            coarse_by_scale[k].add_with_ids(vk, np.array([sid], dtype=np.int64))

        # ---- persist tokens (float16) + mask (uint8) ----
        tok_path  = TOK_DIR / f"n{meta['normal_idx']}_d{meta['depth_idx']}_r{meta['rot_idx']}.fp16.npy"
        mask_path = TOK_DIR / f"n{meta['normal_idx']}_d{meta['depth_idx']}_r{meta['rot_idx']}.mask.npy"
        np.save(tok_path,  t.astype(np.float16))
        np.save(mask_path, mask_flat)

        # ---- manifest row ----
        slice_rows.append({
            "id": sid,
            "token_path": str(tok_path),
            "mask_path":  str(mask_path),

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

        # ---- crash-safe incremental persist ----
        IndexStore.save_faiss(coarse, OUT_DIR / "coarse.faiss")
        for k, idx in coarse_by_scale.items():
            IndexStore.save_faiss(idx, OUT_DIR / f"coarse.scale{k}.faiss")
        write_manifest_slice(slice_rows, OUT_DIR / "manifest.parquet")

    # after building & saving coarse + manifest:
    log("slices", [
        f"Slices        : {len(slice_rows)}",
        f"Global index  : {OUT_DIR/'coarse.faiss'}",
        "Scale indexes : " + ", ".join(str(OUT_DIR / f'coarse.scale{k}.faiss') for k in SLICE_SCALES),
        f"Manifest      : {OUT_DIR/'manifest.parquet'}",
    ])

    # phase-2:
    log("tokens", ["Phase-2 build (IVF-PQ) starting..."])
    create_ivfpq(
        manifest_path=OUT_DIR / "manifest.parquet",
        out_path=TOKEN_INDEX_PATH,
    )
    log("tokens", [f"IVF-PQ index  : {TOKEN_INDEX_PATH}"])


if __name__ == "__main__":
    import multiprocessing as mp
    faiss.omp_set_num_threads(max(1, mp.cpu_count() - 1))
    main()
