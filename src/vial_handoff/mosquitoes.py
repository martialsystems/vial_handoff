# Copyright (c) 2026 Martial Systems LLC
"""Finite mosquito bodies. Stolen drains cargo. Hemolymph kills bodies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vial_handoff.config import RunConfig


@dataclass
class Pool:
    cargo: np.ndarray

    @property
    def n_bodies(self) -> int:
        return int(self.cargo.size)

    @property
    def total_cargo(self) -> float:
        return float(self.cargo.sum()) if self.n_bodies else 0.0


def new_pool(cfg: RunConfig) -> Pool:
    cargo = np.full(int(cfg.m0), float(cfg.cargo_per_body), dtype=np.float64)
    return Pool(cargo=cargo)


def cargo_frac(pool: Pool, cfg: RunConfig) -> float:
    c = pool.total_cargo
    return c / max(c + float(cfg.k_cargo), float(cfg.eps_cargo))


def steal(pool: Pool, demand: float, cfg: RunConfig) -> float:
    """Remove cargo in proportion. demand is sum of trait products * frac."""
    total = pool.total_cargo
    if total <= 0.0 or pool.n_bodies == 0:
        return 0.0
    taken = min(total, max(float(demand), 0.0))
    if taken <= 0.0:
        return 0.0
    scale = 1.0 - taken / total
    pool.cargo *= scale
    return taken


def kill_hemolymph(pool: Pool, hemo_energy_sum: float, cfg: RunConfig, rng: np.random.Generator) -> int:
    """Retire bodies; remaining cargo on those bodies is deleted."""
    if pool.n_bodies == 0:
        return 0
    n_kill = int(np.floor(max(float(hemo_energy_sum), 0.0) / max(float(cfg.soma_energy), 1e-12)))
    n_kill = min(n_kill, pool.n_bodies)
    if n_kill <= 0:
        return 0
    idx = rng.choice(pool.n_bodies, size=n_kill, replace=False)
    keep = np.ones(pool.n_bodies, dtype=bool)
    keep[idx] = False
    pool.cargo = pool.cargo[keep]
    return n_kill


def arrive(pool: Pool, cfg: RunConfig) -> int:
    n = int(cfg.a0)
    if n <= 0:
        return 0
    extra = np.full(n, float(cfg.cargo_per_body), dtype=np.float64)
    pool.cargo = np.concatenate([pool.cargo, extra])
    return n
