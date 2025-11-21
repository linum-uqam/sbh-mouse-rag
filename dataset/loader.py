# dataset/loader.py
from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List

import csv

from volume.volume_helper import (
    AllenVolume,
    NiftiVolume,
    AnnotationHelper,
    VolumeHelper,
    Slice,
)

from .schema import DatasetRow, DatasetSchema, Vec3


class MouseBrainDatasetLoader:
    """
    Loader that uses DatasetSchema + DatasetRow and reconstructs
    slices (Allen + Real) according to:

      - vector (plane normal), depth, rotation
      - crop_* if is_crop == 1

    It does NOT depend on the PNGs themselves — the PNG paths are kept
    only for provenance.
    """

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
        self._allen = AllenVolume(
            cache_dir=str(allen_cache_dir),
            resolution_um=int(allen_resolution_um),
        )
        self._allen.normalize_volume()
        self._annot_helper = (
            AnnotationHelper(
                cache_dir=str(allen_cache_dir),
                resolution_um=int(allen_resolution_um),
            )
            if include_annotation
            else None
        )

        self._real_vol: Optional[NiftiVolume] = None
        if real_volume_path is not None:
            self._real_vol = NiftiVolume(str(Path(real_volume_path).resolve()))
            self._real_vol.normalize_volume()

        # Dataset rows
        self._rows: List[DatasetRow] = []
        self._i = 0  # iterator cursor

        self.load()

    # ---------- ingestion ----------

    def load(self) -> None:
        rows: List[DatasetRow] = []
        with self.csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            DatasetSchema.validate_header(reader.fieldnames)
            for raw in reader:
                rows.append(DatasetSchema.parse_row(raw))
        self._rows = rows
        self.reset()

    # ---------- iteration API ----------

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

    # ---------- random access ----------

    def get(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
          {
            "allen": Slice,
            "real": Optional[Slice],
            "row": DatasetRow,
          }
        """
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

    # ---------- internals ----------

    def _make_slice(
        self,
        vol: VolumeHelper,
        row: DatasetRow,
        *,
        use_own_annotations: bool = False,
    ) -> Slice:
        """
        Build the base slice from the volume, then apply crop_norm if needed.
        """
        base = vol.get_slice(
            normal=row.vector,
            depth=row.depth,
            rotation=row.rotation,
            size=self.size_px,
            pixel=self.pixel_step_vox,
            linear_interp=self.linear_interp,
            include_annotation=self.include_annotation,
            annotation_helper=(None if use_own_annotations else self._annot_helper),
        )

        if row.is_crop:
            base = base.crop_norm(
                cx=row.crop_cx,
                cy=row.crop_cy,
                rw=row.crop_rw,
                rh=row.crop_rh,
                clamp=True,
            )

        return base
