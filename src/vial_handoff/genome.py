# Copyright (c) 2026 Martial Systems LLC
"""Diploid hand-off genome. Six QTLs. No mammal pierce."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vial_handoff.config import RunConfig

FEMALE = 0
MALE = 1

QTL_AUTO = (
    "fruit_use",
    "find_mosquito",
    "cuticle_rasp",
    "gut_probe",
    "digest",
    "heme_safe",
)

I_FRUIT = 0
I_FIND = 1
I_RASP = 2
I_PROBE = 3
I_DIGEST = 4
I_HEME = 5

THIEF_PATH = (I_FIND, I_RASP, I_PROBE)
S_PRE_STARVE = (I_FRUIT,)
S_POST_STARVE = (I_FIND, I_RASP, I_PROBE)


@dataclass
class Pop:
    t: int
    n: int
    ids: np.ndarray
    sex: np.ndarray
    qtl_auto: np.ndarray
    load: np.ndarray
    founder_qtl_auto: np.ndarray
    founder_load: np.ndarray
    mother_id: np.ndarray
    father_id: np.ndarray
    next_id: int
    next_founder: int

    def take(self, idx: np.ndarray) -> Pop:
        if idx.size == 0:
            return empty_pop(
                t=self.t,
                k_a=int(self.qtl_auto.shape[1]),
                n_load=int(self.load.shape[1]),
                next_id=self.next_id,
                next_founder=self.next_founder,
            )
        return Pop(
            t=self.t,
            n=int(idx.size),
            ids=self.ids[idx],
            sex=self.sex[idx],
            qtl_auto=self.qtl_auto[idx],
            load=self.load[idx],
            founder_qtl_auto=self.founder_qtl_auto[idx],
            founder_load=self.founder_load[idx],
            mother_id=self.mother_id[idx],
            father_id=self.father_id[idx],
            next_id=self.next_id,
            next_founder=self.next_founder,
        )


def empty_pop(t: int, k_a: int, n_load: int, next_id: int, next_founder: int) -> Pop:
    return Pop(
        t=t,
        n=0,
        ids=np.empty(0, dtype=np.uint64),
        sex=np.empty(0, dtype=np.uint8),
        qtl_auto=np.empty((0, k_a, 2), dtype=np.float64),
        load=np.empty((0, n_load, 2), dtype=np.uint8),
        founder_qtl_auto=np.empty((0, k_a, 2), dtype=np.uint32),
        founder_load=np.empty((0, n_load, 2), dtype=np.uint32),
        mother_id=np.empty(0, dtype=np.int64),
        father_id=np.empty(0, dtype=np.int64),
        next_id=next_id,
        next_founder=next_founder,
    )


def clip_qtl(arr: np.ndarray, z_max: float) -> np.ndarray:
    return np.clip(arr, -z_max, z_max)


def additive_z(pop: Pop) -> np.ndarray:
    if pop.n == 0:
        return np.empty((0, pop.qtl_auto.shape[1]), dtype=np.float64)
    return pop.qtl_auto.mean(axis=2)
