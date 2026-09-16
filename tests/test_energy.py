# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.fitness import energy_channels, phenotype
from vial_handoff.genome import empty_pop
from vial_handoff.mosquitoes import new_pool


def test_exudate_and_mammal_absent() -> None:
    cfg = RunConfig()
    pool = new_pool(cfg)
    z = np.ones((4, 6), dtype=np.float64)
    fruit, exudate, hemo, stolen, usable, mammal, host, frac, demand, hemo_sum = energy_channels(
        z, False, pool, cfg
    )
    assert np.all(exudate == 0.0)
    assert np.all(mammal == 0.0)
    assert float(stolen.mean()) != float(hemo.mean()) or float(z[0, 3]) == 0.0


def test_gut_probe_raises_stolen_not_hemolymph() -> None:
    cfg = RunConfig()
    pool = new_pool(cfg)
    z = np.ones((2, 6), dtype=np.float64)
    z[1, 3] = 2.0
    _, _, h0, s0, *_ = energy_channels(z[:1], False, pool, cfg)
    _, _, h1, s1, *_ = energy_channels(z[1:2], False, pool, cfg)
    assert float(s1[0]) > float(s0[0])
    assert abs(float(h1[0]) - float(h0[0])) < 1e-12


def test_empty_pop_phenotype() -> None:
    cfg = RunConfig()
    pop = empty_pop(0, 6, 64, 0, 1)
    ph = phenotype(pop, cfg, new_pool(cfg), False)
    assert ph.energy_exudate.size == 0
