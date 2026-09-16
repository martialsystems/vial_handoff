# Copyright (c) 2026 Martial Systems LLC
"""Refuse corpse times. Require a living mid-bridge row in locked JSON."""

from __future__ import annotations

import json
from pathlib import Path

BANNED_T = frozenset({398, 399, 400})
T_SNAP_MAX = 349
REQUIRED = ("rasp", "fluid_detect", "seek")


class SnapshotError(ValueError):
    pass


def load_lock(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_at(lock: dict, t_snap: int) -> dict:
    t_snap = int(t_snap)
    if t_snap >= T_SNAP_MAX + 1:
        raise SnapshotError(f"t_snap {t_snap} >= {T_SNAP_MAX + 1} refused")
    if t_snap in BANNED_T:
        raise SnapshotError(f"t_snap {t_snap} is a last-live corpse")
    for g in lock.get("generations", []):
        if int(g["t"]) == t_snap:
            return g
    raise SnapshotError(f"locked JSON lacks t={t_snap}")


def require_mid_bridge(row: dict, t_snap: int) -> dict:
    n = int(row["n"])
    if n <= 0:
        raise SnapshotError(f"t={t_snap} has n=0")
    if n < 400:
        raise SnapshotError(f"t={t_snap} n={n} is not a high census")
    sw = float(row.get("share_sweat_wound") or 0.0)
    if sw < 0.50:
        raise SnapshotError(f"t={t_snap} sweat+wound {sw} is off")
    qtl = row.get("qtl_mean") or {}
    for name in REQUIRED:
        if name not in qtl:
            raise SnapshotError(f"locked JSON missing qtl_mean.{name}")
    return row
