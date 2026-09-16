# Copyright (c) 2026 Martial Systems LLC
"""Meiosis and mutation. Founder IDs follow allelic lineage."""

from __future__ import annotations

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.fitness import Phenotype, clutch_sizes
from vial_handoff.genome import THIEF_PATH, Pop, clip_qtl, empty_pop
from vial_handoff.mating import Pairing


def _take_hap(arr: np.ndarray, idx: np.ndarray, pick: np.ndarray) -> np.ndarray:
    rows = arr[idx]
    return np.take_along_axis(rows, pick[:, :, None], axis=2)[:, :, 0]


def meiosis_mutate(
    pop: Pop,
    ph: Phenotype,
    pairing: Pairing,
    cfg: RunConfig,
    rng: np.random.Generator,
) -> tuple[Pop, np.ndarray, np.ndarray]:
    k_a, n_load = int(pop.qtl_auto.shape[1]), int(pop.load.shape[1])
    k_found = int(pop.founder_qtl_auto.shape[1])
    if pairing.n_accepted == 0:
        return empty_pop(pop.t + 1, k_a, n_load, pop.next_id, pop.next_founder), np.empty(
            0, dtype=np.int64
        ), np.empty(0, dtype=np.int32)
    fi, mi = pairing.female_idx, pairing.male_idx
    clutch = clutch_sizes(ph.fertility[fi], ph.fertility[mi], cfg)
    n_eggs = int(clutch.sum())
    if n_eggs == 0:
        return empty_pop(pop.t + 1, k_a, n_load, pop.next_id, pop.next_founder), np.empty(
            0, dtype=np.int64
        ), clutch
    mom = np.repeat(fi, clutch)
    dad = np.repeat(mi, clutch)
    sex = rng.integers(0, 2, size=n_eggs, dtype=np.uint8)
    pick_m_a = rng.integers(0, 2, size=(n_eggs, k_a), dtype=np.int64)
    pick_d_a = rng.integers(0, 2, size=(n_eggs, k_a), dtype=np.int64)
    pick_m_f = rng.integers(0, 2, size=(n_eggs, k_found), dtype=np.int64)
    pick_d_f = rng.integers(0, 2, size=(n_eggs, k_found), dtype=np.int64)
    qtl_m = clip_qtl(
        _take_hap(pop.qtl_auto, mom, pick_m_a) + rng.normal(0.0, cfg.sigma_mu, size=(n_eggs, k_a)),
        cfg.z_max,
    )
    qtl_d = clip_qtl(
        _take_hap(pop.qtl_auto, dad, pick_d_a) + rng.normal(0.0, cfg.sigma_mu, size=(n_eggs, k_a)),
        cfg.z_max,
    )
    qtl_auto = np.stack([qtl_m, qtl_d], axis=2)
    founder_qtl_auto = np.stack(
        [_take_hap(pop.founder_qtl_auto, mom, pick_m_f), _take_hap(pop.founder_qtl_auto, dad, pick_d_f)],
        axis=2,
    ).astype(np.uint32)
    rare = rng.random(n_eggs) < cfg.p_rare
    n_rare = int(rare.sum())
    if n_rare:
        host = np.asarray(THIEF_PATH, dtype=np.int64)
        trait = rng.choice(host, size=n_rare)
        hap = rng.integers(0, 2, size=n_rare)
        delta = rng.normal(cfg.rare_mu, cfg.rare_sigma, size=n_rare)
        rows = np.flatnonzero(rare)
        qtl_auto[rows, trait, hap] = clip_qtl(qtl_auto[rows, trait, hap] + delta, cfg.z_max)
    pick_m_l = rng.integers(0, 2, size=(n_eggs, n_load), dtype=np.int64)
    pick_d_l = rng.integers(0, 2, size=(n_eggs, n_load), dtype=np.int64)
    load_m = _take_hap(pop.load, mom, pick_m_l).astype(np.uint8)
    load_d = _take_hap(pop.load, dad, pick_d_l).astype(np.uint8)
    fl_m = _take_hap(pop.founder_load, mom, pick_m_l).astype(np.uint32, copy=True)
    fl_d = _take_hap(pop.founder_load, dad, pick_d_l).astype(np.uint32, copy=True)
    mut_m = (load_m == 0) & (rng.random(load_m.shape) < cfg.u)
    mut_d = (load_d == 0) & (rng.random(load_d.shape) < cfg.u)
    load_m = np.where(mut_m, 1, load_m).astype(np.uint8)
    load_d = np.where(mut_d, 1, load_d).astype(np.uint8)
    n_mut = int(mut_m.sum() + mut_d.sum())
    new_ids = np.arange(pop.next_founder, pop.next_founder + n_mut, dtype=np.uint32)
    cursor = 0
    if mut_m.any():
        k = int(mut_m.sum())
        fl_m[mut_m] = new_ids[cursor : cursor + k]
        cursor += k
    if mut_d.any():
        k = int(mut_d.sum())
        fl_d[mut_d] = new_ids[cursor : cursor + k]
    eggs = Pop(
        t=pop.t + 1,
        n=n_eggs,
        ids=np.arange(pop.next_id, pop.next_id + n_eggs, dtype=np.uint64),
        sex=sex,
        qtl_auto=qtl_auto,
        load=np.stack([load_m, load_d], axis=2),
        founder_qtl_auto=founder_qtl_auto,
        founder_load=np.stack([fl_m, fl_d], axis=2),
        mother_id=pop.ids[mom].astype(np.int64),
        father_id=pop.ids[dad].astype(np.int64),
        next_id=pop.next_id + n_eggs,
        next_founder=pop.next_founder + n_mut,
    )
    return eggs, np.repeat(np.arange(fi.size), clutch), clutch
