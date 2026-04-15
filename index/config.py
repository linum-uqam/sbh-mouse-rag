# index/config.py
from pathlib import Path

# --- Paths ---
OUT_DIR = Path("out/index/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Geometry / sampling ---
K_NORMALS = 64          # Fibonacci normals
SLICE_SIZE = 512        # square slice size in pixels

# Sliding-window patch scales (number of base tiles per side)
# scale = 1  -> full slice
# scale = 2  -> 2x2 base tiles
# scale = 4  -> 4x4 base tiles
# scale = 8  -> 8x8 base tiles
PATCH_SCALES = (1, 2, 4) # ,8

# ------------------------- Geometry (fixed sampling) ------------
FIXED_ROTATIONS = [0.0]
FIXED_PIXEL_STEP_VOX = 1.0
FIXED_STEP_VOX = 2.0
FIXED_MARGIN_VOX = 0.0

# Overlap ratio between neighbouring windows (0.5 = 50% overlap)
PATCH_OVERLAP = 0.25

# --- Embedding dimension (DINOv3 ViT-B/16) ---
D = 768

# --- Index strategy ---
# "auto", "flat", "hnsw", or "ivfpq"
INDEX_STRATEGY = "auto"

# If INDEX_STRATEGY == "auto", we pick:
#   - flat   when #vectors <= AUTO_FLAT_MAX
#   - hnsw   when #vectors <= AUTO_HNSW_MAX
#   - ivfpq  when #vectors >  AUTO_HNSW_MAX
AUTO_FLAT_MAX = 200_000
AUTO_HNSW_MAX = 2_000_000

# --- HNSW params (for cosine/IP) ---
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# --- IVF-PQ params (for cosine/IP) ---
PQ_M = 16
PQ_BITS = 8

