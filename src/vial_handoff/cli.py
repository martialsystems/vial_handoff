# Copyright (c) 2026 Martial Systems LLC
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vial_handoff.config import RunConfig
from vial_handoff.mapping import file_checksum, load_map_spec, map_log, map_population
from vial_handoff.population import run_generations, write_run
from vial_handoff.replay import check_replay, replay_to
from vial_handoff.snapshot import load_lock, require_mid_bridge, row_at

BANNER = "Hand-off: mid-bridge morsus genomes, culex kitchen, no exudate."
REPO = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vial-handoff", description=BANNER)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--from-morsus", type=Path, required=True)
    run.add_argument("--t-snap", type=int, default=200)
    run.add_argument("--map-spec", type=Path, default=REPO / "maps" / "morsus_to_culex.json")
    run.add_argument("--arm", default="knn", choices=["knn", "random"])
    run.add_argument("--generations", type=int, default=2500)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--mosq", type=int, default=200)
    run.add_argument("--arrivals", type=int, default=8)
    run.add_argument("--fruit-forever", action="store_true")
    run.add_argument("--out", type=Path, default=REPO / "logs" / "run.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd != "run":
        return 2
    lock = load_lock(args.from_morsus)
    row = require_mid_bridge(row_at(lock, args.t_snap), args.t_snap)
    spec = load_map_spec(args.map_spec)
    src, rec = replay_to(lock, args.t_snap)
    check_replay(rec, row)
    checksum = file_checksum(args.from_morsus, args.t_snap)
    mate = "random" if args.arm == "random" else "assortative_knn"
    cfg = RunConfig(
        generations=args.generations,
        seed=args.seed,
        t_snap=args.t_snap,
        fruit_forever=bool(args.fruit_forever),
        mating_mode=mate,
        arm=args.arm,
        m0=int(args.mosq),
        a0=int(args.arrivals),
        n_load=int(src.load.shape[1]),
        n=int(src.n),
    )
    rng = np.random.default_rng(cfg.seed)
    pop = map_population(src, cfg, rng, spec)
    pop.t = 0
    mapped = map_log(src, pop)
    result = run_generations(pop, cfg, rng, checksum, jsonl_path=args.out.with_suffix(".jsonl"))
    result["source_lock_row"] = {
        "t": row["t"],
        "n": row["n"],
        "F": row["F"],
        "share_sweat_wound": row.get("share_sweat_wound"),
        "exudate_scale": row.get("exudate_scale"),
    }
    result["map_log"] = mapped
    result["blank_culex_cite"] = "ddb73a8"
    write_run(result, Path(args.out))
    last = result["generations"][-1]
    ev = result["pass_eval"]
    print(
        json.dumps(
            {
                "t": last["t"],
                "n": last["n"],
                "F": last["F"],
                "bodies": last["bodies"],
                "cargo": last["cargo"],
                "share_stolen": last["share_stolen"],
                "share_hemolymph": last["share_hemolymph"],
                "n_killed": last["n_killed_by_hemolymph"],
                "extinct": result["extinct"],
                "pass": ev["pass"],
                "reasons": ev["reasons"],
                "checksum": checksum[:12],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
