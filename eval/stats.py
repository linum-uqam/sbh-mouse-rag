# eval/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math

from eval.metrics import SpatialError


@dataclass
class Stats:
    """
    Lightweight running statistics for the eval loop.
    """

    rows_total: int = 0
    rows_done: int = 0

    # top-1 scores (per row)
    _sum_top1: float = 0.0

    # row-level latency (full pipeline: search + metrics + saving)
    _sum_row_latency_s: float = 0.0

    # pure query latency (only SliceSearcher.search_image)
    _sum_query_latency_s: float = 0.0
    num_queries: int = 0

    # spatial error (per row, averaged over sources)
    rows_with_spatial: int = 0
    _sum_spatial_dist: float = 0.0  # in voxels

    # region error (per row, averaged over sources)
    rows_with_region: int = 0
    _sum_region_mse: float = 0.0

    def update_row(
        self,
        row_latency_s: float,
        query_latencies_s: List[float],
        row_top1_scores: List[float],
        row_spatial_errors: List[SpatialError],
        row_region_errors: List[float],
    ) -> None:
        """
        Update aggregate stats with one dataset row.

        row_latency_s:
            wall-clock time for the entire row (search + metrics + saving).
        query_latencies_s:
            list of per-query latencies (one per call to search_image),
            e.g. [allen_time] or [allen_time, real_time].

        The rest is as before: we average scores/errors per row, not per hit.
        """
        self.rows_done += 1

        # Row-level latency (full pipeline)
        self._sum_row_latency_s += float(row_latency_s)

        # Query-level latency (pure search)
        if query_latencies_s:
            self._sum_query_latency_s += float(sum(query_latencies_s))
            self.num_queries += len(query_latencies_s)

        # Top-1 score (average across sources if multiple)
        if row_top1_scores:
            self._sum_top1 += float(sum(row_top1_scores) / len(row_top1_scores))

        # Spatial distance (voxels)
        spatial_vals = [
            se.dist for se in row_spatial_errors
            if se is not None and not math.isnan(se.dist)
        ]
        if spatial_vals:
            self.rows_with_spatial += 1
            self._sum_spatial_dist += float(sum(spatial_vals) / len(spatial_vals))

        # Region error
        region_vals = [
            e for e in row_region_errors
            if e is not None and not math.isnan(e)
        ]
        if region_vals:
            self.rows_with_region += 1
            self._sum_region_mse += float(sum(region_vals) / len(region_vals))

    # --------- convenience properties ---------
    @property
    def avg_top1(self) -> float:
        return self._sum_top1 / self.rows_done if self.rows_done else float("nan")

    @property
    def avg_row_latency_ms(self) -> float:
        """Average end-to-end time per row (search + metrics + saving)."""
        return (
            1000.0 * self._sum_row_latency_s / self.rows_done
            if self.rows_done
            else float("nan")
        )

    @property
    def avg_query_latency_ms(self) -> float:
        """Average pure search time per query (one call to search_image)."""
        return (
            1000.0 * self._sum_query_latency_s / self.num_queries
            if self.num_queries
            else float("nan")
        )

    # Backwards-compat alias, if anything else still uses avg_latency_ms
    @property
    def avg_latency_ms(self) -> float:
        """Alias for avg_query_latency_ms (pure query time)."""
        return self.avg_query_latency_ms

    @property
    def avg_spatial_dist(self) -> float:
        """Average spatial error in voxels."""
        return (
            self._sum_spatial_dist / self.rows_with_spatial
            if self.rows_with_spatial
            else float("nan")
        )

    @property
    def avg_region_mse(self) -> float:
        """Average region-composition error (0–1)."""
        return (
            self._sum_region_mse / self.rows_with_region
            if self.rows_with_region
            else float("nan")
        )
