# Copyright (c) 2026 Martial Systems LLC
"""Replay locked morsus config to t_snap and dump the living Pop.

The origin JSON has no diploid census. Replay uses vial_morsus as a
library and checks the t_snap row against the lock. Does not write
into the morsus tree.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from vial_morsus.config import RunConfig
from vial_morsus.fitness import phenotype
from vial_morsus.genome import FEMALE, MALE, Pop, init_population
from vial_morsus.inheritance import meiosis_mutate
from vial_morsus.mating import freeze_sigma0, mating_traits, pair
from vial_morsus.metrics import heterozygosity_qtl, record_generation
from vial_morsus.population import cap_uniform, extinct_rule


def config_from_lock(lock: dict) -> RunConfig:
    raw = dict(lock["config"])
    allowed = {f.name for f in fields(RunConfig)}
    raw = {k: v for k, v in raw.items() if k in allowed}
    raw["generations"] = int(lock["config"].get("generations", 2500))
    return RunConfig(**raw)


def replay_to(lock: dict, t_snap: int) -> tuple[Pop, dict]:
    cfg = config_from_lock(lock)
    rng = np.random.default_rng(cfg.seed)
    pop = init_population(cfg, rng)
    t_recover: int | None = None
    ph = phenotype(pop, cfg, t_recover=t_recover)
    h0 = heterozygosity_qtl(pop)
    rec = record_generation(pop, ph, None, cfg, h0, pop.n, pop.n, t_recover)
    ceiling = cfg.ceiling()
    cap_armed = bool(cfg.kinship_cap) and cfg.kinship_cap_on != "recover"
    while pop.t < int(t_snap):
        if extinct_rule(pop, float(ph.survive.mean()) if pop.n else 0.0, cfg):
            raise RuntimeError(f"morsus replay extinct at t={pop.t} before t_snap={t_snap}")
        if (
            not cfg.fruit_forever
            and t_recover is None
            and pop.t >= cfg.t_starve
            and pop.n >= int(cfg.kinship_recover_n)
        ):
            t_recover = int(pop.t)
        if cfg.kinship_cap and cfg.kinship_cap_on == "recover" and not cap_armed and t_recover is not None:
            cap_armed = True
        sigma0 = freeze_sigma0(mating_traits(ph, pop.t, cfg), cfg)
        pairing = pair(pop, ph, cfg, rng, sigma0, cap_armed=cap_armed)
        eggs, _po, _cl = meiosis_mutate(pop, ph, pairing, cfg, rng)
        if eggs.n == 0:
            raise RuntimeError(f"morsus replay no eggs at t={pop.t}")
        host_crowd = pop.n if (pop.t >= cfg.t_starve and not cfg.fruit_forever) else 0
        egg_ph = phenotype(eggs, cfg, t_recover=t_recover, crowd_n=host_crowd)
        survive = (rng.random(eggs.n) < egg_ph.survive) & (egg_ph.v_load > 0.0)
        live = eggs.take(np.flatnonzero(survive))
        pop = cap_uniform(live, ceiling, rng)
        live_crowd = pop.n if (pop.t >= cfg.t_starve and not cfg.fruit_forever) else 0
        ph = phenotype(pop, cfg, t_recover=t_recover, crowd_n=live_crowd)
        rec = record_generation(pop, ph, pairing, cfg, h0, eggs.n, int(survive.sum()), t_recover)
        if rec["extinct"]:
            raise RuntimeError(f"morsus replay extinct at t={pop.t}")
    n_f = int(np.sum(pop.sex == FEMALE))
    n_m = int(np.sum(pop.sex == MALE))
    if n_f == 0 or n_m == 0:
        raise RuntimeError("morsus replay lost a sex")
    return pop, rec


def check_replay(rec: dict, locked_row: dict, *, atol: float = 0.05) -> None:
    if int(rec["n"]) != int(locked_row["n"]):
        raise RuntimeError(f"replay n {rec['n']} != lock n {locked_row['n']}")
    if abs(float(rec["F"]) - float(locked_row["F"])) > atol:
        raise RuntimeError(f"replay F {rec['F']} != lock F {locked_row['F']}")
    for name in ("rasp", "fluid_detect"):
        a = float(rec["qtl_mean"][name])
        b = float(locked_row["qtl_mean"][name])
        if abs(a - b) > atol:
            raise RuntimeError(f"replay {name} {a} != lock {b}")
