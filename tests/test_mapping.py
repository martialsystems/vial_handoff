# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vial_handoff.config import RunConfig
from vial_handoff.genome import I_PROBE, I_RASP
from vial_handoff.mapping import load_map_spec, map_population
from vial_morsus.genome import I_PIERCE as M_PIERCE
from vial_morsus.genome import I_RASP as M_RASP
from vial_morsus.genome import Pop as MorsusPop
from vial_morsus.genome import empty_pop as morsus_empty


def _toy_morsus(n: int = 8) -> MorsusPop:
    rng = np.random.default_rng(0)
    pop = morsus_empty(200, 9, 64, n, n + 10)
    pop.n = n
    pop.ids = np.arange(n, dtype=np.uint64)
    pop.sex = np.array([0, 1] * (n // 2), dtype=np.uint8)
    pop.qtl_auto = rng.normal(0, 0.2, size=(n, 9, 2))
    pop.qtl_auto[:, M_RASP, :] = 1.7
    pop.qtl_auto[:, M_PIERCE, :] = 2.2
    pop.load = np.zeros((n, 64, 2), dtype=np.uint8)
    pop.founder_qtl_auto = np.arange(n * 9 * 2, dtype=np.uint32).reshape(n, 9, 2)
    pop.founder_load = np.arange(n * 64 * 2, dtype=np.uint32).reshape(n, 64, 2)
    pop.mother_id = np.full(n, -1, dtype=np.int64)
    pop.father_id = np.full(n, -1, dtype=np.int64)
    return pop


def test_pierce_does_not_enter_gut_probe() -> None:
    spec = load_map_spec(Path("maps/morsus_to_culex.json"))
    cfg = RunConfig(n_ceiling=1200)
    src = _toy_morsus()
    rng = np.random.default_rng(1)
    dst = map_population(src, cfg, rng, spec)
    assert abs(float(dst.qtl_auto[:, I_RASP, :].mean()) - 1.7) < 1e-9
    assert abs(float(dst.qtl_auto[:, I_PROBE, :].mean()) - 2.2) > 0.5
    assert float(np.max(np.abs(dst.qtl_auto[:, I_PROBE, :]))) < 1.0
    assert "pierce" in spec["not_mapped"]
