# Copyright (c) 2026 Martial Systems LLC
"""Hand-off knobs. Fruit already gone. No mammal bite. No exudate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    n: int = 1000
    generations: int = 2500
    seed: int = 1
    t_snap: int = 200
    fruit_forever: bool = False
    p_female: float = 0.5
    sex_imbalance_max: int = 50
    k_a: int = 6
    n_load: int = 64
    n_lethal: int = 16
    s_let: float = 1.0
    s_sub: float = 0.25
    sigma_init: float = 0.08
    z_max: float = 3.0
    sigma_mu: float = 0.02
    p_rare: float = 0.004
    rare_mu: float = 0.18
    rare_sigma: float = 0.06
    u: float = 0.002
    mating_mode: str = "assortative_knn"
    k: int = 3
    kinship_cap: bool = True
    kinship_cap_on: str = "recover"
    kinship_recover_n: int = 50
    phi_max: float = 0.25
    min_accepted_pairs: int = 8
    m_max: int = 1
    arm: str = "knn"
    cap: str = "off"
    n_floor: int = 8
    n_ceiling: int = 1200
    fail_n_min: int = 8
    fail_viability: float = 0.08
    m0: int = 200
    a0: int = 8
    cargo_per_body: float = 1.0
    k_cargo: float = 80.0
    h0: float = 0.22
    s0: float = 1.20
    soma_energy: float = 1.0
    eps_cargo: float = 1e-9
    thief_threshold: float = 0.08
    eps_stolen: float = 0.002
    eps: float = 0.05
    c0: float = 8.0
    c_quad: float = 0.010
    c_rasp: float = 0.05
    c_probe: float = 0.08
    c_digest: float = 0.40
    c_heme: float = 0.40
    c_heme_in: float = 0.30
    beta_heme: float = 2.0
    heme_rise: float = 0.12
    eps_heme: float = 0.05
    held_stolen_share: float = 0.50
    held_stolen_w: int = 10
    fail_hemolymph_share: float = 0.30
    predator_hemolymph_share: float = 0.50
    fallback_dom_frac: float = 0.20
    survive_steep_fruit: float = 8.0
    survive_thresh_fruit: float = 0.40
    survive_steep_host: float = 250.0
    survive_thresh_host: float = 0.075
    fert_a: float = 1.0
    fert_b: float = 0.35
    fert_d: float = 0.90

    def payload(self) -> dict:
        return asdict(self)

    def ceiling(self) -> int:
        if self.cap == "on":
            return int(self.n)
        return int(self.n_ceiling)
