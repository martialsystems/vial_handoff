# Copyright (c) 2026 Martial Systems LLC
"""Meters. Stolen share is the pass bar. p_thief is secondary."""

from __future__ import annotations

from typing import Any

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.fitness import Phenotype
from vial_handoff.genome import FEMALE, MALE, QTL_AUTO, Pop
from vial_handoff.mating import Pairing
from vial_handoff.mosquitoes import Pool


def _py(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def heterozygosity_qtl(pop: Pop) -> float:
    if pop.n == 0:
        return 0.0
    return float((pop.founder_qtl_auto[:, :, 0] != pop.founder_qtl_auto[:, :, 1]).mean())


def record_generation(
    pop: Pop,
    ph: Phenotype,
    pairing: Pairing | None,
    cfg: RunConfig,
    h0_qtl: float,
    n_eggs: int,
    n_viable: int,
    pool: Pool,
    arrivals: int,
    taken: float,
    n_killed: int,
    checksum: str,
) -> dict:
    n_f = int(np.sum(pop.sex == FEMALE)) if pop.n else 0
    n_m = int(np.sum(pop.sex == MALE)) if pop.n else 0
    h_qtl = heterozygosity_qtl(pop)
    f_t = 0.0 if h0_qtl <= 0 else max(0.0, 1.0 - (h_qtl / h0_qtl))
    qtl_mean: dict[str, float] = {}
    n_names = min(len(QTL_AUTO), ph.z.shape[1] if ph.z.ndim == 2 else 0)
    for i, name in enumerate(QTL_AUTO[:n_names]):
        col = ph.z[:, i] if pop.n else np.empty(0)
        qtl_mean[name] = float(col.mean()) if pop.n else 0.0
    rec = {
        "t": int(pop.t),
        "n": int(pop.n),
        "n_female": n_f,
        "n_male": n_m,
        "F": float(f_t),
        "bodies": int(pool.n_bodies),
        "cargo": float(pool.total_cargo),
        "arrivals": int(arrivals),
        "cargo_taken": float(taken),
        "n_killed_by_hemolymph": int(n_killed),
        "cargo_frac": float(ph.cargo_frac),
        "map_checksum": checksum,
        "qtl_mean": qtl_mean,
        "mean_energy_fruit": float(ph.energy_fruit.mean()) if pop.n else 0.0,
        "mean_energy_exudate": 0.0,
        "mean_energy_hemolymph": float(ph.energy_hemolymph.mean()) if pop.n else 0.0,
        "mean_energy_stolen": float(ph.energy_stolen.mean()) if pop.n else 0.0,
        "mean_usable_stolen": float(ph.usable_stolen.mean()) if pop.n else 0.0,
        "mean_energy_mammal_bite": 0.0,
        "mean_heme_load": float(ph.heme_load.mean()) if pop.n else 0.0,
        "share_fruit": float(ph.share_fruit.mean()) if pop.n else 0.0,
        "share_hemolymph": float(ph.share_hemolymph.mean()) if pop.n else 0.0,
        "share_stolen": float(ph.share_stolen.mean()) if pop.n else 0.0,
        "p_thief": float(np.mean(ph.thief)) if pop.n else 0.0,
        "accepted_pairs": 0 if pairing is None else int(pairing.n_accepted),
        "n_cap_fallback": 0 if pairing is None else int(pairing.n_cap_fallback),
        "n_eggs": int(n_eggs),
        "n_viable": int(n_viable),
        "extinct": bool(pop.n == 0 or n_f == 0 or n_m == 0),
    }
    return {k: _py(v) if not isinstance(v, dict) else v for k, v in rec.items()}


def first_times(records: list[dict], cfg: RunConfig) -> dict:
    t_held_stolen = t_first_thief = None
    t_stolen_load = t_digest_rise = t_heme_safe_rise = None
    hold_run = 0
    hold_start: int | None = None
    w_hold = max(1, int(cfg.held_stolen_w))
    for rec in records:
        t = int(rec["t"])
        if t_first_thief is None and float(rec.get("p_thief", 0.0)) > 0.0:
            t_first_thief = t
        share = float(rec.get("share_stolen", 0.0))
        if share >= cfg.held_stolen_share:
            if hold_run == 0:
                hold_start = t
            hold_run += 1
            if t_held_stolen is None and hold_run >= w_hold:
                t_held_stolen = hold_start
        else:
            hold_run = 0
            hold_start = None
        stolen = float(rec.get("mean_energy_stolen", 0.0))
        digest = float(rec.get("qtl_mean", {}).get("digest", 0.0))
        heme_safe = float(rec.get("qtl_mean", {}).get("heme_safe", 0.0))
        if t_stolen_load is None and stolen > cfg.eps_heme:
            t_stolen_load = t
        if t_digest_rise is None and digest >= cfg.heme_rise:
            t_digest_rise = t
        if t_heme_safe_rise is None and heme_safe >= cfg.heme_rise:
            t_heme_safe_rise = t
    f2500 = next((float(r["F"]) for r in records if int(r["t"]) == 2500), None)
    return {
        "t_first_thief": t_first_thief,
        "t_held_stolen": t_held_stolen,
        "t_stolen_load": t_stolen_load,
        "t_digest_rise": t_digest_rise,
        "t_heme_safe_rise": t_heme_safe_rise,
        "F_2500": f2500,
    }


def evaluate_pass(result: dict, cfg: RunConfig) -> dict:
    recs = result["generations"]
    last = recs[-1]
    live = [r for r in recs if int(r["n"]) > 0]
    last_live = live[-1] if live else last
    t_end = int(last["t"])
    extinct = bool(result.get("extinct"))
    n_fb = int(result.get("n_cap_fallback_gens") or 0)
    fallback_dominated = n_fb > cfg.fallback_dom_frac * float(cfg.generations)
    reasons: list[str] = []
    if extinct:
        reasons.append("extinct")
    if t_end < int(cfg.generations):
        reasons.append("short_run")
    if fallback_dominated:
        reasons.append("fallback_dominated")
    if result.get("t_held_stolen") is None:
        reasons.append("no_held_stolen_share")
    hemo = float(last_live.get("share_hemolymph") or 0.0)
    stolen = float(last_live.get("share_stolen") or 0.0)
    if hemo >= cfg.fail_hemolymph_share:
        reasons.append("hemolymph_ge_0.30")
    if hemo >= cfg.predator_hemolymph_share:
        reasons.append("predator_of_mosquitoes")
    t_load = result.get("t_stolen_load")
    t_dig = result.get("t_digest_rise")
    t_hs = result.get("t_heme_safe_rise")
    if t_dig is not None and (t_load is None or int(t_dig) < int(t_load)):
        reasons.append("digest_before_stolen")
    if t_hs is not None and (t_load is None or int(t_hs) < int(t_load)):
        reasons.append("heme_safe_before_stolen")
    if float(last_live.get("mean_energy_mammal_bite") or 0.0) != 0.0:
        reasons.append("mammal_bite_nonzero")
    if float(last_live.get("mean_energy_exudate") or 0.0) != 0.0:
        reasons.append("exudate_nonzero")
    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "fallback_dominated": fallback_dominated,
        "t_end": t_end,
        "F_end": float(last_live.get("F") or 0.0),
        "share_stolen_end": stolen,
        "share_hemolymph_end": hemo,
        "n_cap_fallback_gens": n_fb,
        "predator": "predator_of_mosquitoes" in reasons,
    }
