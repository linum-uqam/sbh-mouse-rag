# dataset/schema.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import csv
import json

# Type alias for clarity
Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class DatasetRow:
    """
    Canonical dataset row schema.

    Semantics:
      - allen_path / real_path: PNG paths (provenance only; not needed to reconstruct slice)
      - vector: unit normal [x,y,z] for slice plane
      - depth: signed depth along 'vector' in voxel units
      - rotation: in-plane rotation, degrees
      - crop_*: normalized crop params in [0,1]; full slice = cx=cy=0.5, rw=rh=1.0
      - is_crop: 0 for full slice, 1 for crop
      - extra: any non-standard columns in the CSV
    """
    allen_path: str
    real_path: Optional[str]
    vector: Vec3
    depth: float
    rotation: float
    crop_cx: float
    crop_cy: float
    crop_rw: float
    crop_rh: float
    is_crop: int
    extra: Dict[str, Any] = field(default_factory=dict)


class DatasetSchema:
    """
    Single source of truth for CSV layout.

    If you add/remove fields:
      - update COLUMNS
      - update row_to_list(...)
      - update parse_row(...)
    """

    COLUMNS: List[str] = [
        "allen_path",
        "real_path",
        "vector",
        "depth",
        "rotation",
        "crop_cx",
        "crop_cy",
        "crop_rw",
        "crop_rh",
        "is_crop",
    ]

    # ---------- CSV writers ----------

    @classmethod
    def init_csv(cls, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cls.COLUMNS)

    @classmethod
    def append_row(cls, path: Path, row: DatasetRow) -> None:
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(cls.row_to_list(row))

    @classmethod
    def row_to_list(cls, row: DatasetRow) -> List[str]:
        return [
            row.allen_path,
            row.real_path or "",
            json.dumps(list(row.vector)),      # store as "[x,y,z]"
            f"{row.depth}",
            f"{row.rotation}",
            f"{row.crop_cx}",
            f"{row.crop_cy}",
            f"{row.crop_rw}",
            f"{row.crop_rh}",
            f"{int(row.is_crop)}",
        ]

    # ---------- CSV readers ----------

    @classmethod
    def validate_header(cls, fieldnames: Optional[List[str]]) -> None:
        if not fieldnames:
            raise ValueError("CSV has no header")
        missing = set(cls.COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    @classmethod
    def parse_row(cls, raw: Dict[str, str]) -> DatasetRow:
        missing = [c for c in cls.COLUMNS if c not in raw]
        if missing:
            raise ValueError(f"CSV row missing required columns: {missing}")

        allen_path = (raw.get("allen_path") or "").strip()
        real_path = (raw.get("real_path") or "").strip() or None

        vec_str = raw.get("vector") or ""
        vec = json.loads(vec_str)
        if not (isinstance(vec, (list, tuple)) and len(vec) == 3):
            raise ValueError(f"Invalid vector field: {vec_str}")
        vector: Vec3 = (float(vec[0]), float(vec[1]), float(vec[2]))

        def _float(name: str) -> float:
            v = raw.get(name)
            if v is None or v == "":
                raise ValueError(f"Missing required numeric field '{name}'")
            return float(v)

        depth = _float("depth")
        rotation = _float("rotation")
        crop_cx = _float("crop_cx")
        crop_cy = _float("crop_cy")
        crop_rw = _float("crop_rw")
        crop_rh = _float("crop_rh")
        is_crop = int(float(raw.get("is_crop", "0")))

        extra = {k: v for k, v in raw.items() if k not in cls.COLUMNS}

        return DatasetRow(
            allen_path=allen_path,
            real_path=real_path,
            vector=vector,
            depth=depth,
            rotation=rotation,
            crop_cx=crop_cx,
            crop_cy=crop_cy,
            crop_rw=crop_rw,
            crop_rh=crop_rh,
            is_crop=is_crop,
            extra=extra,
        )
