# dataset/data_loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List, Tuple
import csv
import json

from volume.volume_helper import (
    AllenVolume,
    NiftiVolume,
    AnnotationHelper,
    VolumeHelper,
    Slice,
)

# ----------------------------
# Row schema and parsing utils
# ----------------------------

@dataclass(frozen=True)
class DatasetRow:
    allen_path: str                 # kept for provenance; AllenVolume loads from cache
    real_path: Optional[str]
    normal_xyz: Tuple[float, float, float]
    depth_vox: float
    rotation_deg: float
    meta: Dict[str, Any]            # Store original row for traceability (e.g., extra columns)

def _parse_vector(s: str) -> Tuple[float, float, float]:
    v = json.loads(s)  # e.g., "[0.63, 0.02, 0.27]" -> [0.63, 0.02, 0.27]
    if not (isinstance(v, (list, tuple)) and len(v) == 3):
        raise ValueError("Vector must have 3 numbers")
    return (float(v[0]), float(v[1]), float(v[2]))

def _parse_float(s: str, name: str) -> float:
    if s is None or s == "":
        raise ValueError(f"Missing required numeric field '{name}'")
    return float(s)

# ----------------------------
# Public DataLoader
# ----------------------------

class DataLoader:
    def __init__(
        self,
        csv_path: str | Path = "dataset/dataset.csv",
        *,
        allen_cache_dir: str | Path = "volume/data/allen",
        allen_resolution_um: int = 25,
        size_px: int = 512,
        pixel_step_vox: float = 1.0,
        linear_interp: bool = True,
        include_annotation: bool = False,
        real_volume_path: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.size_px = int(size_px)
        self.pixel_step_vox = float(pixel_step_vox)
        self.linear_interp = bool(linear_interp)
        self.include_annotation = bool(include_annotation)

        # Volumes / helpers
        self._allen = AllenVolume(cache_dir=str(allen_cache_dir), resolution_um=int(allen_resolution_um))
        self._allen.normalize_volume()
        self._annot_helper = AnnotationHelper(cache_dir=str(allen_cache_dir), resolution_um=int(allen_resolution_um)) \
                             if include_annotation else None

        # NIfTI volumes
        self._real_vol: Optional[NiftiVolume] = None
        if real_volume_path is not None:
            self._real_vol = NiftiVolume(str(Path(real_volume_path).resolve()))
            self._real_vol.normalize_volume() 

        self._allen_lo_hi = self._allen.get_global_intensity_bounds()
        self._real_lo_hi = self._real_vol.get_global_intensity_bounds() if self._real_vol else None

        # Dataset rows
        self._rows: List[DatasetRow] = []
        self._i = 0  # iterator cursor

        self.load()

    # -------------
    # CSV ingestion
    # -------------
    def load(self) -> None:
        rows: List[DatasetRow] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            self._validate_csv_format(reader)

            for raw in reader:
                allen_path = (raw.get("allen_path") or "").strip()
                real_path = (raw.get("real_path") or "").strip() or None
                normal = _parse_vector(raw.get("vector"))
                depth = _parse_float(raw.get("depth"), "depth")
                rotation = _parse_float(raw.get("rotation"), "rotation")

                row = DatasetRow(
                    allen_path=allen_path,
                    real_path=real_path,
                    normal_xyz=normal,
                    depth_vox=depth,
                    rotation_deg=rotation,
                    meta=dict(raw),  # keep full original row
                )
                rows.append(row)

        self._rows = rows
        self.reset()

    def _validate_csv_format(self, reader):
        required = {"allen_path", "vector", "depth", "rotation"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        
    # -------------
    # Iteration API
    # -------------
    def __len__(self) -> int:
        return len(self._rows)

    def reset(self) -> None:
        self._i = 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.reset()
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._i >= len(self._rows):
            raise StopIteration
        out = self.get(self._i)
        self._i += 1
        return out

    # -------------
    # Random access
    # -------------
    def get(self, idx: int) -> Dict[str, Any]:
        row = self._rows[idx]
        allen_slice = self._make_slice(self._allen, row, use_own_annotations=True)

        real_slice: Optional[Slice] = None
        if self._real_vol is not None:
            real_slice = self._make_slice(self._real_vol, row, use_own_annotations=False)

        return {
            "allen": allen_slice,
            "real": real_slice,
            "row": row,
        }

    # -------------
    # Internals
    # -------------
    def _make_slice(
        self,
        vol: VolumeHelper,
        row: DatasetRow,
        *,
        use_own_annotations: bool = False
    ) -> Slice:
        """
        Generic slice builder for a VolumeHelper.
        If `use_own_annotations=True`, rely on the volume's internal annotation sampler
        (AllenVolume has it). Otherwise, forward the AnnotationHelper so that NIfTI
        volumes (registered to Allen) can get labels too.
        """
        return vol.get_slice(
            normal=row.normal_xyz,
            depth=row.depth_vox,
            rotation=row.rotation_deg,
            size=self.size_px,
            pixel=self.pixel_step_vox,
            linear_interp=self.linear_interp,
            include_annotation=self.include_annotation,
            annotation_helper=(None if use_own_annotations else self._annot_helper),
        )