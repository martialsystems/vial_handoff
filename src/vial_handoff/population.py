# Copyright (c) 2026 Martial Systems LLC
"""Hand-off clock t_h. Fruit already gone. Mosquito pool is not flies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.fitness import phenotype
from vial_handoff.genome import FEMALE, MALE, Pop
from vial_handoff.inheritance import meiosis_mutate
from vial_handoff.mating import freeze_sigma0, mating_traits, pair
from vial_handoff.metrics import evaluate_pass, first_times, heterozygosity_qtl, record_generation
from vial_handoff.mosquitoes import Pool, arrive, kill_hemolymph, new_pool, steal


def cap_uniform(live: Pop, n_cap: int, rng: np.random.Generator) -> Pop:
    if live.n <= n_cap:
        return live
    idx = rng.choice(live.n, size=n_cap, replace=False)
    idx.sort()
    return live.take(idx)


def extinct_rule(pop: Pop, ph_mean_survive: float, cfg: RunConfig) -> bool:
    if pop.n == 0:
        return True
    n_f = int(np.sum(pop.sex == FEMALE))
    n_m = int(np.sum(pop.sex == MALE))
    if n_f == 0 or n_m == 0:
        return True
    if pop.n < cfg.fail_n_min and ph_mean_survive < cfg.fail_viability:
        return True
    return False


def run_generations(
    pop: Pop,
    cfg: RunConfig,
    rng: np.random.Generator,
    checksum: str,
    jsonl_path: Path | None = None,
) -> dict:
    fruit_on = bool(cfg.fruit_forever)
    pool = new_pool(cfg)
    ph = phenotype(pop, cfg, pool, fruit_on)
    h0 = heterozygosity_qtl(pop)
    jsonl_fp = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fp = jsonl_path.open("w", encoding="utf-8")
    rec0 = record_generation(pop, ph, None, cfg, h0, pop.n, pop.n, pool, 0, 0.0, 0, checksum)
    records = [rec0]
    if jsonl_fp is not None:
        jsonl_fp.write(json.dumps(rec0) + "\n")
        jsonl_fp.flush()
    ceiling = cfg.ceiling()
    cap_armed = bool(cfg.kinship_cap) and (cfg.kinship_cap_on != "recover" or pop.n >= cfg.kinship_recover_n)
    n_fallback_gens = 0
    for _step in range(cfg.generations):
        if extinct_rule(pop, float(ph.survive.mean()) if pop.n else 0.0, cfg):
            records[-1]["extinct"] = True
            break
        if cfg.kinship_cap and not cap_armed and pop.n >= int(cfg.kinship_recover_n):
            cap_armed = True
        sigma0 = freeze_sigma0(mating_traits(ph, cfg), cfg)
        pairing = pair(pop, ph, cfg, rng, sigma0, cap_armed=cap_armed)
        n_fallback_gens += int(pairing.n_cap_fallback)
        taken = steal(pool, ph.steal_demand, cfg)
        n_killed = kill_hemolymph(pool, ph.hemo_sum, cfg, rng)
        n_arr = arrive(pool, cfg)
        eggs, _po, _cl = meiosis_mutate(pop, ph, pairing, cfg, rng)
        n_eggs = eggs.n
        if n_eggs == 0:
            pop = eggs
            ph = phenotype(pop, cfg, pool, fruit_on)
            rec = record_generation(pop, ph, pairing, cfg, h0, 0, 0, pool, n_arr, taken, n_killed, checksum)
            rec["extinct"] = True
            records.append(rec)
            break
        egg_ph = phenotype(eggs, cfg, pool, fruit_on)
        survive = (rng.random(eggs.n) < egg_ph.survive) & (egg_ph.v_load > 0.0)
        n_viable = int(survive.sum())
        live = eggs.take(np.flatnonzero(survive))
        pop = cap_uniform(live, ceiling, rng)
        ph = phenotype(pop, cfg, pool, fruit_on)
        rec = record_generation(pop, ph, pairing, cfg, h0, n_eggs, n_viable, pool, n_arr, taken, n_killed, checksum)
        records.append(rec)
        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps(rec) + "\n")
            jsonl_fp.flush()
        if rec["extinct"] or extinct_rule(pop, float(ph.survive.mean()) if pop.n else 0.0, cfg):
            records[-1]["extinct"] = True
            break
    if jsonl_fp is not None:
        jsonl_fp.close()
    last = records[-1]
    result = {
        "config": cfg.payload(),
        "seed": cfg.seed,
        "arm": cfg.arm,
        "mate": "random" if cfg.mating_mode == "random" else "knn",
        "k": cfg.k,
        "t_snap": cfg.t_snap,
        "map_checksum": checksum,
        "h0_qtl": h0,
        "generations": records,
        "final_t": last["t"],
        "extinct": bool(last["extinct"]),
        "n_cap_fallback_gens": n_fallback_gens,
        **first_times(records, cfg),
    }
    result["pass_eval"] = evaluate_pass(result, cfg)
    return result


def write_run(result: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
