from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, PathPatch, Ellipse
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from PIL import Image, ImageOps


# =========================
# CONFIG — edit paths here
# =========================
CONFIG = {
    "query_img": "scripts/paper/img/query_rot0.png",

    # If None -> auto-generate 0/90/180/270 from query_img.
    # Otherwise provide 4 paths [0, 90, 180, 270].
    "rotations_imgs": [
        "scripts/paper/img/query_rot0.png",
        "scripts/paper/img/query_rot90.png",
        "scripts/paper/img/query_rot180.png",
        "scripts/paper/img/query_rot270.png",
    ],

    "volume_img": "scripts/paper/img/3dbrain.PNG",

    "tranches_imgs": [
        "scripts/paper/img/y.PNG",
        "scripts/paper/img/z.PNG",
        "scripts/paper/img/v.PNG",
        "scripts/paper/img/x.PNG",
    ],

    "topk_imgs": [
        "scripts/paper/img/hit_01_pid11399_score0.8833.png",
        "scripts/paper/img/hit_02_pid1137_score0.8783.png",
        "scripts/paper/img/hit_03_pid10549_score0.8764.png",
    ],

    "convert_to_grayscale": True,
    "out_prefix": "scripts/paper/img/pipeline_these_frozen_trainable_custom",
    "dpi_png": 300,

    # 0.0 = no borders (recommended)
    "img_outline_lw": 0.0,
    "img_outline_color": "#888888",
    
    # Standard fire icon (PNG with alpha recommended)
    "trainable_icon_path": "scripts/paper/img/icon_fire.png",

    # Icon sizing in "data units" (tune once and you're done)
    "trainable_icon_size_box": 0.24,   # icon size inside MLP box
    "trainable_icon_size_leg": 0.18,   # icon size in legend

    # Fallback emoji if the png is missing (handy during setup)
    "trainable_icon_emoji": "🔥",
}


def _load_icon_rgba(path: str | Path | None) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing icon: {p}")
        return None
    im = Image.open(p).convert("RGBA")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def draw_icon_rgba(ax, center: Tuple[float, float], size: float, icon_rgba: np.ndarray, z=8):
    """
    Draw an RGBA icon centered at `center` with width=height=`size` in data coords.
    """
    cx, cy = center
    half = size / 2
    ax.imshow(
        icon_rgba,
        extent=(cx - half, cx + half, cy - half, cy + half),
        interpolation="nearest",
        zorder=z,
    )

# =========================
# Image utilities
# =========================
def _load_image(path: str | Path, grayscale: bool = True) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing image: {path}")
        return None
    im = Image.open(path)
    if grayscale:
        im = im.convert("L")
        arr = np.asarray(im).astype(np.float32) / 255.0
        return arr
    im = im.convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def _fit_square(arr: np.ndarray, out_size: int = 256) -> np.ndarray:
    if arr.ndim == 2:
        im = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    else:
        im = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")

    im2 = ImageOps.fit(
        im,
        (out_size, out_size),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    return np.asarray(im2).astype(np.float32) / 255.0


def _rotate90(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, k=k)


# =========================
# Drawing primitives
# =========================
def round_box(ax, x, y, w, h, fc="#ffffff", ec="#222222", lw=1.2, r=0.10, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, s, **kw):
    ax.text(x, y, s, **kw)


def arrow(ax, p0, p1, color="#222222", lw=1.2, style="-|>", ms=12, conn="arc3", z=3):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(
            arrowstyle=style,
            lw=lw,
            color=color,
            mutation_scale=ms,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=conn,
        ),
        zorder=z,
    )


def snowflake_icon(ax, center, r=0.085, color="#1f77b4", lw=1.4):
    cx, cy = center
    for k in range(6):
        ang = k * np.pi / 3
        x0, y0 = cx - r * np.cos(ang), cy - r * np.sin(ang)
        x1, y1 = cx + r * np.cos(ang), cy + r * np.sin(ang)
        ax.add_line(
            Line2D([x0, x1], [y0, y1], lw=lw, color=color, solid_capstyle="round", zorder=6)
        )
        for s in (-1, 1):
            ang2 = ang + s * np.pi / 8
            bx0, by0 = cx + 0.45 * r * np.cos(ang), cy + 0.45 * r * np.sin(ang)
            bx1, by1 = bx0 + 0.35 * r * np.cos(ang2), by0 + 0.35 * r * np.sin(ang2)
            ax.add_line(
                Line2D([bx0, bx1], [by0, by1], lw=lw * 0.8, color=color, solid_capstyle="round", zorder=6)
            )


def flame_icon(ax, center, size=0.22, color="#ff7f0e", ec="#c95b00", lw=1.0):
    """
    Standard-looking flame icon (vector), with an inner tongue.
    - Stable in PDF/SVG (no emoji/font dependency).
    """
    cx, cy = center
    s = size / 2.0

    # Outer flame: wide base, narrow top, symmetric, "icon-like"
    verts = [
        (0.00, -1.00),  # bottom tip
        (0.70, -0.80), (0.95, -0.10), (0.40,  0.15),   # right lower lobe
        (0.75,  0.55), (0.40,  0.95), (0.05,  1.10),   # right upper -> near tip
        (0.00,  1.20), (-0.05,  1.10), (-0.40, 0.95),  # tip -> left upper
        (-0.75, 0.55), (-0.40, 0.15), (-0.95,-0.10),   # left mid
        (-0.70,-0.80), (-0.25,-1.05), (0.00,-1.00),    # close to bottom
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    ]
    outer = [(cx + s * x, cy + s * y) for x, y in verts]
    ax.add_patch(PathPatch(MplPath(outer, codes), facecolor=color, edgecolor=ec, lw=lw,
                           zorder=6, joinstyle="round"))

    # Inner flame tongue (teardrop)
    verts2 = [
        (0.00, -0.35),
        (0.35, -0.20), (0.45,  0.20), (0.18,  0.35),
        (0.30,  0.60), (0.12,  0.85), (0.00,  0.95),
        (-0.12, 0.85), (-0.30, 0.60), (-0.18, 0.35),
        (-0.45, 0.20), (-0.35,-0.20), (0.00,-0.35),
    ]
    codes2 = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    ]
    inner = [(cx + s * x, cy + s * y) for x, y in verts2]
    ax.add_patch(PathPatch(MplPath(inner, codes2), facecolor="#ffd08a", edgecolor="none",
                           lw=0, zorder=7))

def vector_bar(ax, x, y, w=0.16, h=1.18):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#ffffff", edgecolor="#333333", lw=1.1, zorder=2))
    for t in np.linspace(y + 0.17, y + h - 0.17, 5):
        ax.add_line(Line2D([x + 0.03, x + w - 0.04], [t, t], lw=1.0, color="#333333", zorder=3))


# Borderless image rendering (optional outline via cfg)
def draw_img(
    ax,
    x,
    y,
    w,
    h,
    img: np.ndarray,
    z=2,
    grayscale=True,
    pad=0.0,
    outline_lw: float = 0.0,
    outline_ec: str = "#888888",
):
    cmap = "gray" if grayscale and img.ndim == 2 else None
    ax.imshow(
        img,
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(x + pad, x + w - pad, y + pad, y + h - pad),
        interpolation="nearest",
        zorder=z,
    )
    if outline_lw and outline_lw > 0:
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=outline_ec, linewidth=outline_lw, zorder=z + 1)
        )


def draw_overlapping_tiles(
    ax,
    x,
    y,
    tile=0.60,
    dx=0.10,
    dy=0.08,
    imgs=None,
    grayscale=True,
    outline_lw: float = 0.0,
    outline_ec: str = "#888888",
):
    if imgs is None or len(imgs) != 4:
        base = np.clip(np.linspace(0.2, 0.9, 256)[None, :].repeat(256, 0), 0, 1)
        imgs = [base, base, base, base]

    for i in range(3, -1, -1):
        xi = x + (3 - i) * dx
        yi = y + (3 - i) * dy
        draw_img(
            ax,
            xi,
            yi,
            tile,
            tile,
            imgs[i],
            z=2 + i,
            grayscale=grayscale,
            pad=0.0,
            outline_lw=outline_lw,
            outline_ec=outline_ec,
        )

    return (x, y, tile + 3 * dx, tile + 3 * dy)


def draw_topk_stack_no_box(ax, x, y_center, imgs: Sequence[np.ndarray], tile=0.48, gap=0.05, grayscale=True):
    assert len(imgs) == 3, "topk must have 3 images"

    ell_h = 0.06
    lab_h = 0.14
    h = 3 * tile + 2 * gap + ell_h + lab_h

    y_top = y_center + h / 2
    y_img_top = y_top - tile

    cmap = "gray" if grayscale else None
    for i in range(3):
        yi = y_img_top - i * (tile + gap)
        ax.imshow(
            imgs[i],
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=(x, x + tile, yi, yi + tile),
            interpolation="nearest",
            zorder=1,
        )

    y_third_bottom = y_img_top - 2 * (tile + gap)
    y_ell = y_third_bottom - 0.04
    label(ax, x + tile / 2, y_ell, "", ha="center", va="top", fontsize=13.5, color="#666666")
    label(ax, x + tile / 2, y_ell - 0.10, r"Top-$k$", ha="center", va="top", fontsize=10, color="#1a1a1a")


def draw_database_icon(ax, x, y, w, h, fc="#ffffff", ec="#444444", lw=1.2, z=2):
    # ellipse caps height
    eh = min(h * 0.24, w * 0.55)

    # Body
    ax.add_patch(
        Rectangle((x, y + eh / 2), w, h - eh, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    )

    # Top cap
    ax.add_patch(
        Ellipse((x + w / 2, y + h - eh / 2), w, eh, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z + 0.2)
    )

    # Bottom cap
    ax.add_patch(
        Ellipse((x + w / 2, y + eh / 2), w, eh, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z + 0.1)
    )


# =========================
# Build figure (v13 layout)
# =========================
def build_figure(cfg: dict) -> Tuple[plt.Figure, plt.Axes]:
    grayscale = bool(cfg.get("convert_to_grayscale", True))
    outline_lw = float(cfg.get("img_outline_lw", 0.0))
    outline_ec = str(cfg.get("img_outline_color", "#888888"))
    train_icon = _load_icon_rgba(cfg.get("trainable_icon_path"))
    train_emoji = str(cfg.get("trainable_icon_emoji", "🔥"))
    train_size_box = float(cfg.get("trainable_icon_size_box", 0.24))
    train_size_leg = float(cfg.get("trainable_icon_size_leg", 0.18))

    # Load & prep images
    q = _load_image(cfg["query_img"], grayscale=grayscale)
    if q is None:
        q = np.clip(np.linspace(0.2, 0.9, 256)[None, :].repeat(256, 0), 0, 1)
    q = _fit_square(q, out_size=256)

    # Rotations
    rot_paths = cfg.get("rotations_imgs", None)
    if rot_paths is None:
        rots = [_rotate90(q, k) for k in (0, 1, 2, 3)]
    else:
        rots = []
        for p in rot_paths:
            im = _load_image(p, grayscale=grayscale)
            if im is None:
                im = q
            rots.append(_fit_square(im, out_size=256))
        if len(rots) != 4:
            raise ValueError("rotations_imgs must have 4 paths (0/90/180/270).")

    vol = _load_image(cfg["volume_img"], grayscale=grayscale)
    if vol is None:
        vol = q
    vol = _fit_square(vol, out_size=256)

    tr_paths = cfg["tranches_imgs"]
    if len(tr_paths) != 4:
        raise ValueError("tranches_imgs must contain 4 paths.")
    tr = []
    for p in tr_paths:
        im = _load_image(p, grayscale=grayscale)
        if im is None:
            im = q
        tr.append(_fit_square(im, out_size=256))

    tk_paths = cfg["topk_imgs"]
    if len(tk_paths) != 3:
        raise ValueError("topk_imgs must contain 3 paths.")
    tk = []
    for p in tk_paths:
        im = _load_image(p, grayscale=grayscale)
        if im is None:
            im = q
        tk.append(_fit_square(im, out_size=256))

    # Canvas
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.15)
    ax.axis("off")

    # Colors
    c_text = "#1a1a1a"
    c_line = "#222222"
    c_lane_top = "#f7f7f7"
    c_lane_bot = "#f2f2f2"
    c_frozen_fc = "#e8f2ff"
    c_frozen_ec = "#1f77b4"
    c_train_fc = "#fff0e0"
    c_train_ec = "#ff7f0e"
    c_db_ec = "#444444"

    # Lanes
    lane_x0, lane_w = 0.35, 12.70
    top_y0, top_h = 3.75, 2.30
    bot_y0, bot_h = 1.35, 2.30
    ax.add_patch(Rectangle((lane_x0, top_y0), lane_w, top_h, facecolor=c_lane_top, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((lane_x0, bot_y0), lane_w, bot_h, facecolor=c_lane_bot, edgecolor="none", zorder=0))

    label(ax, lane_x0 + 0.15, top_y0 + top_h - 0.10, "Recherche", ha="left", va="top",
          fontsize=12, color="#666666", fontweight="bold")
    label(ax, lane_x0 + 0.15, bot_y0 + bot_h - 0.10, "Pré-calcul", ha="left", va="top",
          fontsize=12, color="#666666", fontweight="bold")

    y_rt = top_y0 + top_h / 2 - 0.10
    y_pc = bot_y0 + bot_h / 2 - 0.10
    y_mid = (y_rt + y_pc) / 2

    # X layout
    q_x = 0.55
    q_w = 1.10  # square
    q_h = 1.10  # square
    st_x = 2.45
    enc_x, enc_w = 3.85, 2.00
    bar_x = 6.45
    cos_x, cos_w = 7.55, 1.85
    mlp_x, mlp_w = 10.05, 1.85
    top_x = 12.45

    # Query image (square)
    q_y = y_rt - q_h / 2
    draw_img(ax, q_x, q_y, q_w, q_h, q, grayscale=grayscale, outline_lw=outline_lw, outline_ec=outline_ec)
    label(ax, q_x + q_w / 2, q_y - 0.16, "Requête", ha="center", va="top", fontsize=10, color=c_text)

    # Rotations tiles
    tile_rot = 0.60
    rot_y = y_rt - (tile_rot + 3 * 0.08) / 2
    bb_rot = draw_overlapping_tiles(
        ax, st_x, rot_y, tile=tile_rot, dx=0.10, dy=0.08,
        imgs=rots, grayscale=grayscale, outline_lw=outline_lw, outline_ec=outline_ec
    )
    label(ax, st_x + bb_rot[2] / 2, rot_y - 0.16, "Rotations", ha="center", va="top", fontsize=10, color=c_text)
    arrow(ax, (q_x + q_w, y_rt), (st_x, y_rt), color=c_line, lw=1.2, conn="arc3")

    # Volume image
    v_h = 1.10
    v_y = y_pc - v_h / 2
    draw_img(ax, q_x, v_y, q_w, v_h, vol, grayscale=grayscale, outline_lw=outline_lw, outline_ec=outline_ec)
    label(ax, q_x + q_w / 2, v_y - 0.16, "Volume 3D", ha="center", va="top", fontsize=10, color=c_text)

    # n tranches tiles
    slice_y = y_pc - (tile_rot + 3 * 0.08) / 2
    bb_s = draw_overlapping_tiles(
        ax, st_x, slice_y, tile=tile_rot, dx=0.10, dy=0.08,
        imgs=tr, grayscale=grayscale, outline_lw=outline_lw, outline_ec=outline_ec
    )
    label(ax, st_x + bb_s[2] / 2, slice_y - 0.16, r"$n$ tranches", ha="center", va="top", fontsize=10, color=c_text)
    arrow(ax, (q_x + q_w, y_pc), (st_x, y_pc), color=c_line, lw=1.2, conn="arc3")

    # Encoder
    enc_y, enc_h = y_mid - 0.50, 1.00
    round_box(ax, enc_x, enc_y, enc_w, enc_h, fc=c_frozen_fc, ec=c_frozen_ec, lw=1.7)
    label(ax, enc_x + enc_w / 2, enc_y + enc_h / 2, "Encodeur\nDINO",
          ha="center", va="center", fontsize=10, fontweight="bold")
    snowflake_icon(ax, (enc_x + enc_w - 0.24, enc_y + 0.24), r=0.085, color=c_frozen_ec, lw=1.4)

    # Curved arrows into encoder
    arrow(ax, (st_x + bb_rot[2], y_rt), (enc_x, enc_y + enc_h * 0.78), color=c_line, lw=1.2, conn="arc3,rad=-0.18")
    arrow(ax, (st_x + bb_s[2], y_pc), (enc_x, enc_y + enc_h * 0.22), color=c_line, lw=1.2, conn="arc3,rad=0.18")

    # Cosine + FAISS
    cos_h = 0.86
    cos_y = y_rt - cos_h / 2

    # Make FAISS cylinder narrower + taller than cosine
    fa_w = cos_w * 0.65
    fa_h = cos_h * 1.85
    x_mid = cos_x + cos_w / 2
    fa_x = x_mid - fa_w / 2
    fa_y = y_pc - fa_h / 2

    # Cosine box with formula
    round_box(ax, cos_x, cos_y, cos_w, cos_h, fc="#ffffff", ec="#333333", lw=1.2)
    label(
        ax,
        cos_x + cos_w / 2,
        cos_y + cos_h / 2,
        r"$\cos(\mathbf{q},\mathbf{i})=\frac{\mathbf{q}\cdot\mathbf{i}}{\|\mathbf{q}\|\;\|\mathbf{i}\|}$",
        ha="center",
        va="center",
        fontsize=10.0,
        color=c_text,
    )

    # FAISS as cylinder (no internal lines)
    draw_database_icon(ax, fa_x, fa_y, fa_w, fa_h, fc="#ffffff", ec=c_db_ec, lw=1.2, z=2)
    label(ax, fa_x + fa_w / 2, fa_y + fa_h / 2, "Index\nFAISS", ha="center", va="center",
          fontsize=10, fontweight="bold")

    # Embedding vectors
    bar_w, bar_h = 0.16, 1.18
    bar_rt_y = (cos_y + cos_h / 2) - bar_h / 2
    bar_pc_y = (fa_y + fa_h / 2) - bar_h / 2

    vector_bar(ax, bar_x, bar_rt_y, w=bar_w, h=bar_h)
    vector_bar(ax, bar_x, bar_pc_y, w=bar_w, h=bar_h)
    label(ax, bar_x + bar_w / 2, bar_rt_y - 0.12, "768 dim.", ha="center", va="top", fontsize=8.8, color="#555555")
    label(ax, bar_x + bar_w / 2, bar_pc_y - 0.12, "768 dim.", ha="center", va="top", fontsize=8.8, color="#555555")

    enc_right = (enc_x + enc_w, enc_y + enc_h / 2)
    arrow(ax, enc_right, (bar_x, bar_rt_y + bar_h / 2), color=c_line, lw=1.2, conn="arc3")
    arrow(ax, enc_right, (bar_x, bar_pc_y + bar_h / 2), color=c_line, lw=1.2, conn="arc3")

    arrow(ax, (bar_x + bar_w, bar_rt_y + bar_h / 2), (cos_x, cos_y + cos_h / 2), color=c_line, lw=1.2, conn="arc3")
    arrow(ax, (bar_x + bar_w, bar_pc_y + bar_h / 2), (fa_x, fa_y + fa_h / 2), color=c_line, lw=1.2, conn="arc3")

    # FAISS <-> cosine (double arrow) aligned on cosine center x
    ax.annotate(
        "",
        xy=(x_mid, cos_y),
        xytext=(x_mid, fa_y + fa_h),
        arrowprops=dict(
            arrowstyle="<->",
            lw=1.2,
            color="#555555",
            linestyle=(0, (4, 3)),
            mutation_scale=11,
        ),
        zorder=3,
    )

    # Reranker
    c_train_fc = "#fff0e0"
    c_train_ec = "#ff7f0e"
    mlp_y, mlp_h = cos_y, cos_h
    round_box(ax, mlp_x, mlp_y, mlp_w, mlp_h, fc=c_train_fc, ec=c_train_ec, lw=1.8)
    label(ax, mlp_x + mlp_w / 2, mlp_y + mlp_h / 2, "Réordonneur\nMLP",
          ha="center", va="center", fontsize=10, fontweight="bold")
    icon_center = (mlp_x + mlp_w - 0.25, mlp_y + 0.25)
    if train_icon is not None:
        draw_icon_rgba(ax, icon_center, train_size_box, train_icon, z=9)
    else:
        ax.text(icon_center[0], icon_center[1], train_emoji,
                ha="center", va="center", fontsize=14, zorder=9)
        
    # cosine -> mlp (horizontal)
    arrow(ax, (cos_x + cos_w, y_rt), (mlp_x, y_rt), color=c_line, lw=1.2, conn="arc3")

    # Top-k
    draw_topk_stack_no_box(ax, top_x, y_center=y_rt, imgs=tk, tile=0.48, gap=0.05, grayscale=grayscale)
    arrow(ax, (mlp_x + mlp_w, y_rt), (top_x, y_rt), color=c_line, lw=1.2, conn="arc3")

    # Legend
    leg_y = 0.80
    leg_x1 = 7.55
    snowflake_icon(ax, (leg_x1, leg_y), r=0.062, color=c_frozen_ec, lw=1.2)
    label(ax, leg_x1 + 0.18, leg_y, "module figé", fontsize=9.5, color=c_text, va="center", ha="left")

    leg_x2 = leg_x1 + 2.20
    icon_center = (leg_x2, leg_y)
    if train_icon is not None:
        draw_icon_rgba(ax, icon_center, train_size_leg, train_icon, z=9)
    else:
        ax.text(icon_center[0], icon_center[1], train_emoji,
                ha="center", va="center", fontsize=13, zorder=9)
    label(ax, leg_x2 + 0.18, leg_y, "module entraînable", fontsize=9.5, color=c_text, va="center", ha="left")

    return fig, ax


def main() -> None:
    fig, _ = build_figure(CONFIG)

    out_prefix = Path(CONFIG.get("out_prefix", "pipeline"))
    dpi_png = int(CONFIG.get("dpi_png", 300))

    pdf_path = out_prefix.with_suffix(".pdf")
    svg_path = out_prefix.with_suffix(".svg")
    png_path = out_prefix.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(png_path, dpi=dpi_png, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    print(f"[OK] Wrote:\n  {pdf_path}\n  {svg_path}\n  {png_path}")


if __name__ == "__main__":
    main()