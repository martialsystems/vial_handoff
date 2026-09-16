# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

from pathlib import Path

from vial_handoff.config import RunConfig
from vial_handoff.genome import QTL_AUTO

REPO = Path(__file__).resolve().parents[1]


def test_defaults() -> None:
    cfg = RunConfig()
    assert cfg.t_snap == 200
    assert cfg.m0 == 200
    assert cfg.a0 == 8
    assert cfg.h0 == 0.22
    assert cfg.s0 == 1.20
    assert "pierce" not in QTL_AUTO
    assert "sweat" not in " ".join(QTL_AUTO)


def test_no_graphforge() -> None:
    assert not (REPO / "engine_pin.json").exists()
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    assert "Do not pin GraphForge" in agents
    long_arm = (REPO / "LONG_ARM.md").read_text(encoding="utf-8")
    assert "Curiosity is not a transition." in long_arm
