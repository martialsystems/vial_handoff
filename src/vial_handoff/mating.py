# Copyright (c) 2026 Martial Systems LLC
"""k-NN on {find_mosquito, cuticle_rasp, gut_probe}. Delayed cap. Fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.fitness import Phenotype
from vial_handoff.genome import FEMALE, MALE, S_POST_STARVE, Pop


@dataclass
class Pairing:
    female_idx: np.ndarray
    male_idx: np.ndarray
    distance: np.ndarray
    n_failed_match: int
    n_accepted: int
    n_kinship_reject: int = 0
    n_cap_fallback: int = 0


def mating_traits(ph: Phenotype, cfg: RunConfig) -> np.ndarray:
    cols = list(S_POST_STARVE)
    if ph.z.size == 0:
        return np.empty((0, len(cols)), dtype=np.float64)
    return ph.z[:, cols]


def freeze_sigma0(traits: np.ndarray, cfg: RunConfig) -> np.ndarray:
    floor = max(float(cfg.sigma_init), 1e-6)
    n_col = int(traits.shape[1]) if traits.ndim == 2 else 1
    if traits.ndim != 2 or traits.shape[0] <= 1:
        return np.full(n_col, floor, dtype=np.float64)
    sd = traits.std(axis=0, ddof=1)
    return np.where(sd <= 1e-12, floor, sd).astype(np.float64)


def pairwise_fm_distance(z_f: np.ndarray, z_m: np.ndarray, sigma0: np.ndarray) -> np.ndarray:
    delta = (z_f[:, None, :] - z_m[None, :, :]) / sigma0[None, None, :]
    return np.sqrt(np.square(delta).sum(axis=2))


def pairwise_fm_phi(founder_f: np.ndarray, founder_m: np.ndarray) -> np.ndarray:
    n_f, k, _ = founder_f.shape
    n_m = int(founder_m.shape[0])
    if n_f == 0 or n_m == 0 or k == 0:
        return np.zeros((n_f, n_m), dtype=np.float64)
    acc = np.zeros((n_f, n_m), dtype=np.float64)
    for loc in range(k):
        mat_f, pat_f = founder_f[:, loc, 0], founder_f[:, loc, 1]
        mat_m, pat_m = founder_m[:, loc, 0], founder_m[:, loc, 1]
        acc += (mat_f[:, None] == mat_m[None, :]).astype(np.float64)
        acc += (mat_f[:, None] == pat_m[None, :]).astype(np.float64)
        acc += (pat_f[:, None] == mat_m[None, :]).astype(np.float64)
        acc += (pat_f[:, None] == pat_m[None, :]).astype(np.float64)
    return acc * (0.25 / k)


def _empty_pair(n_fail: int) -> Pairing:
    return Pairing(
        female_idx=np.empty(0, dtype=np.int64),
        male_idx=np.empty(0, dtype=np.int64),
        distance=np.empty(0, dtype=np.float64),
        n_failed_match=int(n_fail),
        n_accepted=0,
    )


def _pair_once(
    pop: Pop,
    ph: Phenotype,
    cfg: RunConfig,
    rng: np.random.Generator,
    sigma0: np.ndarray,
    apply_cap: bool,
    force_random: bool,
) -> Pairing:
    f_idx = np.flatnonzero(pop.sex == FEMALE)
    m_idx = np.flatnonzero(pop.sex == MALE)
    if f_idx.size == 0 or m_idx.size == 0:
        return _empty_pair(int(f_idx.size))
    z = mating_traits(ph, cfg)
    dist = pairwise_fm_distance(z[f_idx], z[m_idx], sigma0)
    remaining = np.full(m_idx.size, int(cfg.m_max), dtype=np.int32)
    order = rng.permutation(f_idx.size)
    knn_mode = (not force_random) and cfg.mating_mode != "random"
    k_eff = min(int(cfg.k), int(m_idx.size))
    knn = None
    if knn_mode and k_eff > 0 and not apply_cap:
        knn = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
    phi_fm = None
    if apply_cap:
        phi_fm = pairwise_fm_phi(pop.founder_qtl_auto[f_idx], pop.founder_qtl_auto[m_idx])
    chosen_f: list[int] = []
    chosen_m: list[int] = []
    chosen_d: list[float] = []
    n_fail = 0
    n_kin_rej = 0
    for local_f in order:
        if apply_cap:
            cap = remaining > 0
            kin_ok = phi_fm[local_f] <= cfg.phi_max
            n_kin_rej += int(np.sum(cap & ~kin_ok))
            legal = np.flatnonzero(cap & kin_ok)
            if legal.size == 0:
                n_fail += 1
                continue
            if knn_mode:
                k_loc = min(int(cfg.k), int(legal.size))
                take = legal[np.argpartition(dist[local_f, legal], kth=k_loc - 1)[:k_loc]]
                pick = int(take[np.argmin(dist[local_f, take])])
            else:
                pick = int(rng.choice(legal))
        elif knn_mode:
            cands = knn[local_f]
            legal = cands[remaining[cands] > 0]
            if legal.size == 0:
                n_fail += 1
                continue
            pick = int(legal[np.argmin(dist[local_f, legal])])
        else:
            cap = remaining > 0
            if not np.any(cap):
                n_fail += 1
                continue
            pick = int(rng.choice(np.flatnonzero(cap)))
        chosen_f.append(int(f_idx[local_f]))
        chosen_m.append(int(m_idx[pick]))
        chosen_d.append(float(dist[local_f, pick]))
        remaining[pick] -= 1
    if not chosen_f:
        empty = _empty_pair(n_fail)
        empty.n_kinship_reject = n_kin_rej
        return empty
    return Pairing(
        female_idx=np.asarray(chosen_f, dtype=np.int64),
        male_idx=np.asarray(chosen_m, dtype=np.int64),
        distance=np.asarray(chosen_d, dtype=np.float64),
        n_failed_match=n_fail,
        n_accepted=len(chosen_f),
        n_kinship_reject=n_kin_rej,
    )


def pair(
    pop: Pop,
    ph: Phenotype,
    cfg: RunConfig,
    rng: np.random.Generator,
    sigma0: np.ndarray,
    cap_armed: bool = False,
) -> Pairing:
    apply = bool(cfg.kinship_cap and cap_armed)
    first = _pair_once(pop, ph, cfg, rng, sigma0, apply_cap=apply, force_random=False)
    if apply and first.n_accepted < int(cfg.min_accepted_pairs):
        fb = _pair_once(pop, ph, cfg, rng, sigma0, apply_cap=False, force_random=True)
        fb.n_cap_fallback = 1
        fb.n_kinship_reject = first.n_kinship_reject
        return fb
    return first
