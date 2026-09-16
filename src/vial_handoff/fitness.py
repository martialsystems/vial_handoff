# Copyright (c) 2026 Martial Systems LLC
"""Stolen midgut blood vs hemolymph. Exudate and mammal bite are 0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.genome import I_DIGEST, I_FIND, I_FRUIT, I_HEME, I_PROBE, I_RASP, Pop, additive_z
from vial_handoff.mosquitoes import Pool, cargo_frac


def pos(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, np.inf)


def logistic(energy: np.ndarray, thresh: float, steep: float) -> np.ndarray:
    x = np.clip(steep * (energy - thresh), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_weights(load: np.ndarray, cfg: RunConfig) -> np.ndarray:
    if load.size == 0:
        return np.ones((0, load.shape[1] if load.ndim == 3 else 0), dtype=np.float64)
    hom = (load[:, :, 0] == 1) & (load[:, :, 1] == 1)
    w = np.ones(load.shape[:2], dtype=np.float64)
    n_let = int(cfg.n_lethal)
    w[:, :n_let] = np.where(hom[:, :n_let], 1.0 - cfg.s_let, 1.0)
    w[:, n_let:] = np.where(hom[:, n_let:], 1.0 - cfg.s_sub, 1.0)
    return w


@dataclass
class Phenotype:
    z: np.ndarray
    energy_fruit: np.ndarray
    energy_exudate: np.ndarray
    energy_hemolymph: np.ndarray
    energy_stolen: np.ndarray
    usable_stolen: np.ndarray
    energy_mammal_bite: np.ndarray
    energy_host: np.ndarray
    energy: np.ndarray
    heme_load: np.ndarray
    v_iron: np.ndarray
    cost: np.ndarray
    v_load: np.ndarray
    survive: np.ndarray
    fertility: np.ndarray
    w: np.ndarray
    thief: np.ndarray
    steal_demand: float
    hemo_sum: float
    cargo_frac: float
    share_fruit: np.ndarray
    share_hemolymph: np.ndarray
    share_stolen: np.ndarray


def energy_channels(z: np.ndarray, fruit_on: bool, pool: Pool, cfg: RunConfig):
    n = int(z.shape[0])
    fruit = pos(z[:, I_FRUIT]) if fruit_on else np.zeros(n)
    find = pos(z[:, I_FIND])
    rasp = pos(z[:, I_RASP])
    probe = pos(z[:, I_PROBE])
    frac = cargo_frac(pool, cfg)
    hemo = find * rasp * float(cfg.h0)
    stolen = find * rasp * probe * frac * float(cfg.s0)
    usable = stolen * sigmoid(z[:, I_DIGEST])
    exudate = np.zeros(n)
    mammal = np.zeros(n)
    host = usable + hemo
    return fruit, exudate, hemo, stolen, usable, mammal, host, frac, float((find * rasp * probe * frac).sum()), float(hemo.sum())


def phenotype(pop: Pop, cfg: RunConfig, pool: Pool, fruit_on: bool) -> Phenotype:
    z = additive_z(pop)
    empty = np.empty(0, dtype=np.float64)
    if pop.n == 0:
        return Phenotype(
            z=z,
            energy_fruit=empty,
            energy_exudate=empty,
            energy_hemolymph=empty,
            energy_stolen=empty,
            usable_stolen=empty,
            energy_mammal_bite=empty,
            energy_host=empty,
            energy=empty,
            heme_load=empty,
            v_iron=empty,
            cost=empty,
            v_load=empty,
            survive=empty,
            fertility=empty,
            w=empty,
            thief=np.empty(0, dtype=bool),
            steal_demand=0.0,
            hemo_sum=0.0,
            cargo_frac=0.0,
            share_fruit=empty,
            share_hemolymph=empty,
            share_stolen=empty,
        )
    fruit, exudate, hemo, stolen, usable, mammal, host, frac, demand, hemo_sum = energy_channels(
        z, fruit_on, pool, cfg
    )
    heme_load = float(cfg.c_heme_in) * stolen
    v_iron = np.exp(-cfg.beta_heme * heme_load / (1.0 + pos(z[:, I_HEME])))
    empty_pool = 1.0 if pool.n_bodies == 0 else 0.0
    unused = (stolen < cfg.eps_stolen).astype(np.float64)
    cost = (
        cfg.c_quad * np.square(pos(z)).sum(axis=1)
        + cfg.c_rasp * pos(z[:, I_RASP]) * empty_pool
        + cfg.c_probe * pos(z[:, I_PROBE]) * empty_pool
        + cfg.c_digest * pos(z[:, I_DIGEST]) * unused
        + cfg.c_heme * pos(z[:, I_HEME]) * unused
    )
    v_load = load_weights(pop.load, cfg).prod(axis=1)
    if fruit_on:
        steep, thresh = cfg.survive_steep_fruit, cfg.survive_thresh_fruit
    else:
        steep, thresh = cfg.survive_steep_host, cfg.survive_thresh_host
    total = fruit + host
    survive = logistic(total, thresh=thresh, steep=steep) * v_iron
    fertility = pos(cfg.fert_a + cfg.fert_b * total - cfg.fert_d * cost) * v_load
    w = survive * fertility * np.exp(-cost) * v_load
    den = np.maximum(total, 1e-12)
    return Phenotype(
        z=z,
        energy_fruit=fruit,
        energy_exudate=exudate,
        energy_hemolymph=hemo,
        energy_stolen=stolen,
        usable_stolen=usable,
        energy_mammal_bite=mammal,
        energy_host=host,
        energy=total,
        heme_load=heme_load,
        v_iron=v_iron,
        cost=cost,
        v_load=v_load,
        survive=survive,
        fertility=fertility,
        w=w,
        thief=stolen > cfg.thief_threshold,
        steal_demand=demand,
        hemo_sum=hemo_sum,
        cargo_frac=frac,
        share_fruit=fruit / den,
        share_hemolymph=hemo / den,
        share_stolen=usable / den,
    )


def clutch_sizes(fert_f: np.ndarray, fert_m: np.ndarray, cfg: RunConfig) -> np.ndarray:
    return np.maximum(0, np.rint(cfg.c0 * fert_f * fert_m)).astype(np.int32)
