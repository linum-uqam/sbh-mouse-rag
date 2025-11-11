from pathlib import Path

# ------------------------- Output paths -------------------------
OUT_DIR = Path("index/index")
TOK_DIR = OUT_DIR / "tokens"

# ------------------------- Embedding dims -----------------------
D = 768                 # embedding dimension
SLICE_SIZE = 512        # sampled slice size (pixels)
K_NORMALS = 64          # number of normals (Fibonacci sampling)

# ------------------------- FAISS: global / per-scale ------------
USE_HNSW = False        # True -> HNSW-IP, False -> Flat-IP
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

# Always build multi-scale pooled slice vectors (no legacy mode)
SLICE_SCALES = [2, 4, 8, 14]   # pooled token grids; 1x1 is implicit in "coarse.faiss"

# ------------------------- FAISS: token IVF-PQ ------------------
TOKEN_INDEX_PATH = OUT_DIR / "tokens.ivfpq.faiss"
IVF_NLIST = 4096            # number of coarse lists
PQ_M = 64                    # number of PQ subvectors
PQ_BITS = 8                  # bits per PQ subvector

# ------------------------- Geometry (fixed sampling) ------------
FIXED_ROTATIONS = [0.0]
FIXED_PIXEL_STEP_VOX = 1.0
FIXED_STEP_VOX = 10.0
FIXED_MARGIN_VOX = 0.0
# ---------------------------------------------------------------
