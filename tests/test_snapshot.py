# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

from pathlib import Path

import pytest

from vial_handoff.snapshot import SnapshotError, load_lock, require_mid_bridge, row_at

MORSUS = Path("/Users/samw/vial_morsus/logs/knn_2500_s1.json")


def test_refuses_t_ge_350() -> None:
    if not MORSUS.is_file():
        pytest.skip("morsus lock missing")
    lock = load_lock(MORSUS)
    with pytest.raises(SnapshotError, match="refused"):
        row_at(lock, 350)
    with pytest.raises(SnapshotError):
        row_at(lock, 400)


def test_requires_t_200() -> None:
    if not MORSUS.is_file():
        pytest.skip("morsus lock missing")
    lock = load_lock(MORSUS)
    with pytest.raises(SnapshotError):
        row_at(lock, 1999)
    row = require_mid_bridge(row_at(lock, 200), 200)
    assert int(row["n"]) >= 400
    assert float(row["share_sweat_wound"]) >= 0.5
    assert "rasp" in row["qtl_mean"]
    assert "fluid_detect" in row["qtl_mean"]
