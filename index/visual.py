# index/visual.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw

from volume.volume_helper import AllenVolume


__all__ = [
    "OverlayParams",
    "save_png",
    "draw_heat_overlay",
    "save_results_images",
]


# ------------------------- Public API -------------------------

@dataclass(frozen=True)
class OverlayParams:
    """Parameters for heat overlay rendering."""
    alpha_max: float = 0.65          # max alpha blend for hottest cells
    clip_low_q: float = 0.60         # lower percentile for contrast stretch
    clip_high_q: float = 0.98        # upper percentile for contrast stretch
    gamma: float = 0.6               # gamma <1 boosts peaks
    draw_grid: bool = False          # draw grid lines
    topk: int = 20                   # outline top-K hottest cells (0 to disable)
    grid_color: Tuple[int, int, int] = (80, 80, 80)
    box_color: Tuple[int, int, int] = (255, 255, 255)


def save_png(array01: np.ndarray, out_path: Path) -> None:
    """Save a float array in [0,1] as PNG."""
    arr = (np.asarray(array01) * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path)


def draw_heat_overlay(
    img01: np.ndarray,
    heat_1d: List[float],
    grid_h: int,
    grid_w: int,
    params: OverlayParams = OverlayParams(),
) -> Image.Image:
    """
    Build a color-mapped, alpha-blended overlay on top of a grayscale base image.
    - Contrast normalized by percentiles + gamma
    - Per-cell alpha ∝ normalized heat
    - Optional grid lines and top-K boxes
    Returns a PIL.Image (RGB).
    """
    base = Image.fromarray(
        (np.asarray(img01) * 255.0).clip(0, 255).astype(np.uint8),
        mode="L"
    ).convert("RGB")
    W, H = base.size

    heat = np.array(heat_1d, dtype=np.float32).reshape(int(grid_h), int(grid_w))
    norm = _normalize_heat(heat, params.clip_low_q, params.clip_high_q, params.gamma)

    # Prepare per-cell color & alpha
    rgb_grid = [[(0, 0, 0)] * grid_w for _ in range(grid_h)]
    a_grid = [[0.0] * grid_w for _ in range(grid_h)]
    for i in range(grid_h):
        for j in range(grid_w):
            v = float(norm[i, j])
            rgb_grid[i][j] = _colormap_inferno_lite(v)
            a_grid[i][j] = v * params.alpha_max

    # Paint
    overlay = base.copy()
    px = overlay.load()
    cell_w = W / grid_w
    cell_h = H / grid_h

    for i in range(grid_h):
        y0 = int(round(i * cell_h))
        y1 = int(round((i + 1) * cell_h))
        for j in range(grid_w):
            x0 = int(round(j * cell_w))
            x1 = int(round((j + 1) * cell_w))
            r, g, b = rgb_grid[i][j]
            a = a_grid[i][j]
            if a <= 0.0:
                continue
            for y in range(y0, y1):
                for x in range(x0, x1):
                    br, bg, bb = px[x, y]
                    px[x, y] = (
                        int((1 - a) * br + a * r),
                        int((1 - a) * bg + a * g),
                        int((1 - a) * bb + a * b),
                    )

    if params.draw_grid:
        _draw_grid(overlay, grid_h, grid_w, params.grid_color)

    if params.topk and params.topk > 0:
        _draw_topk_boxes(overlay, norm, params.topk, params.box_color)

    return overlay


def save_results_images(hits: List[Dict], save_dir: Path, overlay_params: OverlayParams = OverlayParams()) -> None:
    """
    Save top-K matched slices as PNGs using exact sampling params stored in hits.
    Files:
      - rank_01_id_<slice_id>.png
      - rank_01_id_<slice_id>__col.png (if 'col_heat' present)
    """
    if not hits:
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    allen = AllenVolume(
        cache_dir="volume/data/allen",
        resolution_um=int(hits[0]["resolution_um"])
    )

    for h in hits:
        sid = int(h["slice_id"])
        rank = int(h["rank"])

        slc = allen.get_slice(
            normal=tuple(map(float, h["normal"])),
            depth=float(h["depth_vox"]),
            rotation=float(h["rotation_deg"]),
            size=int(h["size_px"]),
            pixel=1.0,
            linear_interp=bool(h["linear_interp"]),
            include_annotation=False,
        )

        if hasattr(allen, "is_valid_slice") and not allen.is_valid_slice(slc):
            continue

        img01 = slc.normalized().image

        base_path = save_dir / f"rank_{rank:02d}_id_{sid}.png"
        save_png(img01, base_path)

        if "col_heat" in h and h["col_heat"] is not None:
            grid_h = int(h["grid_h"])
            grid_w = int(h["grid_w"])
            overlay = draw_heat_overlay(img01, h["col_heat"], grid_h, grid_w, params=overlay_params)
            overlay_path = save_dir / f"rank_{rank:02d}_id_{sid}__col.png"
            overlay.save(overlay_path)


# ------------------------- Internal helpers -------------------------

def _normalize_heat(heat: np.ndarray, qlow: float, qhigh: float, gamma: float) -> np.ndarray:
    """Percentile clip → [0,1] → gamma."""
    lo = float(np.quantile(heat, qlow))
    hi = float(np.quantile(heat, qhigh))
    if hi <= lo:
        hi = float(heat.max())
        lo = float(heat.min())
    if hi > lo:
        norm = (heat - lo) / (hi - lo)
    else:
        norm = np.zeros_like(heat)
    norm = np.clip(norm, 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    return norm


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _colormap_inferno_lite(v01: float) -> Tuple[int, int, int]:
    """Lightweight inferno-like ramp: dark→purple→orange→yellow."""
    stops = [
        (0.0,  (0, 0, 0)),
        (0.25, (31, 12, 72)),
        (0.50, (102, 37, 103)),
        (0.75, (237, 121, 83)),
        (1.0,  (252, 255, 164)),
    ]
    v = max(0.0, min(1.0, float(v01)))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if v <= t1:
            if t1 == t0:
                return c1
            t = (v - t0) / (t1 - t0)
            r = int(_lerp(c0[0], c1[0], t))
            g = int(_lerp(c0[1], c1[1], t))
            b = int(_lerp(c0[2], c1[2], t))
            return (r, g, b)
    return stops[-1][1]


def _draw_grid(img: Image.Image, grid_h: int, grid_w: int, color: Tuple[int, int, int]) -> None:
    """Draw cell grid lines over the image."""
    W, H = img.size
    cell_w = W / grid_w
    cell_h = H / grid_h
    draw = ImageDraw.Draw(img)
    # horizontal lines
    for i in range(1, grid_h):
        y = int(round(i * cell_h))
        draw.line([(0, y), (W, y)], fill=color, width=1)
    # vertical lines
    for j in range(1, grid_w):
        x = int(round(j * cell_w))
        draw.line([(x, 0), (x, H)], fill=color, width=1)


def _draw_topk_boxes(img: Image.Image, norm_heat: np.ndarray, topk: int, color: Tuple[int, int, int]) -> None:
    """Draw top-K hottest cell rectangles (ignore zero-heat/background cells)."""
    W, H = img.size
    grid_h, grid_w = norm_heat.shape
    cell_w = W / grid_w
    cell_h = H / grid_h

    flat = norm_heat.reshape(-1)
    if flat.size == 0 or topk <= 0:
        return

    eps = 1e-6
    pos_idx = np.flatnonzero(flat > eps)
    if pos_idx.size == 0:
        return

    k = min(int(topk), int(pos_idx.size))
    # argpartition within the positive set
    top_local = np.argpartition(-flat[pos_idx], k - 1)[:k]
    idx = pos_idx[top_local]

    yy, xx = np.unravel_index(idx, (grid_h, grid_w))
    draw = ImageDraw.Draw(img)
    for i, j in zip(yy.tolist(), xx.tolist()):
        x0 = int(round(j * cell_w)); x1 = int(round((j + 1) * cell_w))
        y0 = int(round(i * cell_h)); y1 = int(round((i + 1) * cell_h))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
