# Copyright (c) 2026 Martial Systems LLC
"""Lossy morsus -> hand-off map. pierce does not enter gut_probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.genome import I_DIGEST, I_FIND, I_FRUIT, I_HEME, I_PROBE, I_RASP, Pop, clip_qtl
from vial_morsus.genome import I_DIGEST as M_DIGEST
from vial_morsus.genome import I_FLUID as M_FLUID
from vial_morsus.genome import I_FRUIT as M_FRUIT
from vial_morsus.genome import I_HEME as M_HEME
from vial_morsus.genome import I_RASP as M_RASP
from vial_morsus.genome import I_SEEK as M_SEEK
from vial_morsus.genome import Pop as MorsusPop


def file_checksum(path: Path, t_snap: int) -> str:
    raw = Path(path).read_bytes()
    h = hashlib.sha256()
    h.update(raw)
    h.update(b"|t_snap=")
    h.update(str(int(t_snap)).encode("ascii"))
    return h.hexdigest()


def load_map_spec(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def map_population(
    src: MorsusPop,
    cfg: RunConfig,
    rng: np.random.Generator,
    spec: dict,
) -> Pop:
    if "rasp" not in spec["copy_alleles"] or "fluid_detect" not in spec["find_mosquito"]:
        raise ValueError("map spec missing rasp or fluid_detect")
    n = int(src.n)
    if n > int(cfg.n_ceiling):
        idx = np.sort(rng.choice(n, size=int(cfg.n_ceiling), replace=False))
        src = src.take(idx)
        n = int(src.n)
    k_a = int(cfg.k_a)
    qtl = np.zeros((n, k_a, 2), dtype=np.float64)
    qtl[:, I_FRUIT, :] = src.qtl_auto[:, M_FRUIT, :]
    qtl[:, I_RASP, :] = src.qtl_auto[:, M_RASP, :]
    qtl[:, I_DIGEST, :] = src.qtl_auto[:, M_DIGEST, :]
    qtl[:, I_HEME, :] = src.qtl_auto[:, M_HEME, :]
    find = 0.5 * src.qtl_auto[:, M_FLUID, :] + 0.5 * src.qtl_auto[:, M_SEEK, :]
    qtl[:, I_FIND, :] = find
    qtl[:, I_PROBE, :] = rng.normal(0.0, float(spec["new_draw"]["gut_probe"]["sd"]), size=(n, 2))
    qtl = clip_qtl(qtl, float(spec.get("z_max", cfg.z_max)))
    return Pop(
        t=0,
        n=n,
        ids=src.ids.copy(),
        sex=src.sex.copy(),
        qtl_auto=qtl,
        load=src.load.copy(),
        founder_qtl_auto=src.founder_qtl_auto.copy(),
        founder_load=src.founder_load.copy(),
        mother_id=src.mother_id.copy(),
        father_id=src.father_id.copy(),
        next_id=src.next_id,
        next_founder=src.next_founder,
    )


def map_log(src: MorsusPop, dst: Pop) -> dict:
    z_s = src.qtl_auto.mean(axis=2)
    z_d = dst.qtl_auto.mean(axis=2)
    return {
        "n": int(dst.n),
        "mean_rasp_src": float(z_s[:, M_RASP].mean()),
        "mean_cuticle_rasp": float(z_d[:, I_RASP].mean()),
        "mean_fluid_src": float(z_s[:, M_FLUID].mean()),
        "mean_seek_src": float(z_s[:, M_SEEK].mean()),
        "mean_find_mosquito": float(z_d[:, I_FIND].mean()),
        "mean_pierce_src": float(z_s[:, 3].mean()),
        "mean_gut_probe": float(z_d[:, I_PROBE].mean()),
        "pierce_not_mapped": True,
    }
