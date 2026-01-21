# eval/stats.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import math


@dataclass
class Stats:
    rows_total: int = 0
    rows_done: int = 0

    _sum_top1: float = 0.0
    _sum_row_latency_s: float = 0.0

    _sum_query_latency_s: float = 0.0
    num_queries: int = 0

    rows_with_geom: int = 0
    _sum_geom_dist: float = 0.0

    # top-1 corner chamfer in um (per row, averaged across sources)
    rows_with_corner: int = 0
    _sum_corner_um: float = 0.0

    def update_row(
        self,
        *,
        row_latency_s: float,
        query_latencies_s: List[float],
        row_top1_scores: List[float],
        row_top1_geom_dists: List[float],
        row_top1_corner_um: List[float],
    ) -> None:
        self.rows_done += 1
        self._sum_row_latency_s += float(row_latency_s)

        if query_latencies_s:
            self._sum_query_latency_s += float(sum(query_latencies_s))
            self.num_queries += len(query_latencies_s)

        if row_top1_scores:
            self._sum_top1 += float(sum(row_top1_scores) / len(row_top1_scores))

        geom_vals = [d for d in row_top1_geom_dists if d is not None and math.isfinite(d)]
        if geom_vals:
            self.rows_with_geom += 1
            self._sum_geom_dist += float(sum(geom_vals) / len(geom_vals))

        corner_vals = [c for c in row_top1_corner_um if c is not None and math.isfinite(c)]
        if corner_vals:
            self.rows_with_corner += 1
            self._sum_corner_um += float(sum(corner_vals) / len(corner_vals))

    @property
    def avg_top1(self) -> float:
        return self._sum_top1 / self.rows_done if self.rows_done else float("nan")

    @property
    def avg_row_latency_ms(self) -> float:
        return (1000.0 * self._sum_row_latency_s / self.rows_done) if self.rows_done else float("nan")

    @property
    def avg_query_latency_ms(self) -> float:
        return (1000.0 * self._sum_query_latency_s / self.num_queries) if self.num_queries else float("nan")

    @property
    def avg_geom_dist(self) -> float:
        return (self._sum_geom_dist / self.rows_with_geom) if self.rows_with_geom else float("nan")

    @property
    def avg_corner_um(self) -> float:
        return (self._sum_corner_um / self.rows_with_corner) if self.rows_with_corner else float("nan")
