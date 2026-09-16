# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict | None:
    path = REPO / "logs" / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_knn_handoff_predator_fail() -> None:
    found = [_load(f"handoff_2500_s{s}.json") for s in (1, 2, 3)]
    if any(r is None for r in found):
        pytest.skip("handoff JSON not generated")
    for run in found:
        assert run is not None
        assert run["t_snap"] == 200
        assert int(run["source_lock_row"]["n"]) == 1200
        assert float(run["source_lock_row"]["share_sweat_wound"]) >= 0.5
        assert run["pass_eval"]["pass"] is False
        assert "predator_of_mosquitoes" in run["pass_eval"]["reasons"]
        last = run["generations"][-1]
        assert int(last["t"]) == 2500
        assert float(last["share_hemolymph"]) >= 0.50
        assert float(last["mean_energy_exudate"]) == 0.0
        assert float(last["mean_energy_mammal_bite"]) == 0.0
        assert run["blank_culex_cite"] == "ddb73a8"
        assert run["map_log"]["pierce_not_mapped"] is True
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "0 of 3" in readme
    assert "Fail-predator" in readme or "fail-predator" in readme
    assert "ddb73a8" in readme
    assert "69f4f6d" in readme


def test_fruit_forever_stolen_clean() -> None:
    run = _load("fruit_forever_400_s1.json")
    if run is None:
        pytest.skip("fruit-forever JSON not generated")
    last = run["generations"][-1]
    assert int(last["t"]) == 400
    assert float(last["share_stolen"]) < 0.05
    assert run["extinct"] is False
