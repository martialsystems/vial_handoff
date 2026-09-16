# vial_handoff

If we take living mid-bridge morsus k-NN flies (already past fruit-off, already rasping films) and put them in the culex kitchen (finite engorged mosquitoes, no mammal skin, no exudate), can they add a gut probe and live on stolen midgut blood?

No. 0 of 3 hand-off seeds. They lived on hemolymph. Stolen share never held. gut_probe stayed untrained. This is fail-predator, not a culex-cliff replay. Blank-founder culex remains `ddb73a8` / science `fa819c7`. Do not average. Science lock `e843b8c`.

Parents: [vial_morsus](https://github.com/martialsystems/vial_morsus) `69f4f6d` at t_snap=200 (n=1,200, sweat+wound about 0.98). [vial_culex](https://github.com/martialsystems/vial_culex) kitchen, exudate off.

## Origin

Mapped full living census at t=200. rasp copied to cuticle_rasp. find_mosquito = 0.5 fluid_detect + 0.5 seek. pierce not mapped. gut_probe ~ N(0, 0.08). Fruit already gone. M0=200 bodies, a0=8. Hemolymph meals kill bodies.

Copied from `logs/handoff_2500_s{1,2,3}.json`, `logs/fruit_forever_400_s1.json`, `logs/random_2500_s1.json`.

| seed | mate | morsus F@200 | map rasp | map find | map probe | t_h=2500 n | stolen | hemolymph | bodies | killed/gen | gut_probe end | rasp end | PASS |
|-----:|------|-------------:|---------:|---------:|----------:|-----------:|-------:|----------:|-------:|-----------:|--------------:|---------:|:----:|
| 1 | knn | 0.235 | 1.000 | 0.343 | -0.001 | 1,200 | 0.000001 | 1.000 | 8 | 8 | -0.475 | 2.905 | no |
| 2 | knn | 0.218 | 0.800 | 0.376 | -0.004 | 1,200 | 0 | 1.000 | 8 | 8 | -0.386 | 2.909 | no |
| 3 | knn | 0.230 | 0.833 | 0.281 | 0.003 | 1,200 | 0.000067 | 1.000 | 8 | 8 | -0.278 | 2.913 | no |
| 1 | fruit-forever | 0.235 | 1.000 | 0.343 | -0.001 | 1,200 at t=400 | 0.00081 | 0.111 | 8 | 8 | -0.046 | 1.326 | control |
| 1 | random | 0.235 | 1.000 | 0.343 | -0.001 | 1,200 | 0.000011 | 1.000 | 8 | 8 | -0.908 | 2.854 | contrast |

Film-rasp flies became mosquito predators (rasp × find). They did not add a gut probe. Fruit-forever stolen share 0.00081 < 0.05. PASS needs >=2 of 3 knn seeds. 0 of 3 is a fail. Do not raise gut_probe payoff. Do not restamp morsus or culex.

Halt. No 10k.

## How to run

```text
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" -e /path/to/vial_morsus
.venv/bin/python -m pytest
.venv/bin/python -m vial_handoff run --from-morsus data/morsus_locks/knn_2500_s1.json \
  --t-snap 200 --map-spec maps/morsus_to_culex.json \
  --mosq 200 --arrivals 8 --generations 2500 --seed 1 \
  --out logs/handoff_2500_s1.json
```

## Files

| Path | Role |
|------|------|
| `src/vial_handoff/` | snapshot, replay, mapping, mosquitoes, fitness, mating, population, cli |
| `maps/morsus_to_culex.json` | lossy trait map |
| `data/morsus_locks/` | read-only copies of morsus k-NN origin JSON |
| `AGENTS.md` | laws, claim bans, VBD |
| `LONG_ARM.md` | halt; next legal node: none |
| `logs/handoff_2500_s{1,2,3}.json` | hand-off knn locks |

Do not pin GraphForge. Do not restamp `69f4f6d` or `ddb73a8`.

[Fly research index](https://gist.github.com/martialsystems/12835f747d6360781f3cc7f91f243178)
