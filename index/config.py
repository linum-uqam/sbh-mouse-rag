from pathlib import Path

# ------------------------- Config -------------------------
OUT_DIR = Path("index/index")
TOK_DIR = OUT_DIR / "tokens"

# geometry / encoder
K_NORMALS = 64
SLICE_SIZE = 512
D = 768

# faiss
USE_HNSW = False            # flip to True if you want HNSW instead of Flat
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# geometry 
FIXED_ROTATIONS = [0.0]
FIXED_PIXEL_STEP_VOX = 1.0
FIXED_STEP_VOX = 10.0
FIXED_MARGIN_VOX = 0.0
# ----------------------------------------------------------

