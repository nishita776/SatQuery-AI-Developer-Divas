# SatQuery AI

Day 0 skeleton (BE-00). Read the Master Document and your handover first: the contract code in
`satquery/contracts/` wins over any prose.

## Setup (Python 3.11 recommended; 3.10-3.12 work)

```bash
python3.11 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[geo,server,dev]"
pre-commit install
python fixtures/make_fixtures.py        # only needed if fixtures/scenes is missing or you changed the generator
```

## The commands you will use every day

```bash
SATQUERY_MODE=fake pytest tests/contract tests/e2e tests/p2     # everything (about 15 s)
SATQUERY_MODE=fake pytest tests/contract tests/e2e              # the merge gate (F3.8)
python fixtures/make_fixtures.py                                # regenerate fixtures + fake results + stub stories
ruff check . && ruff format .
```

## Environment variables

| Variable | Meaning |
|---|---|
| `SATQUERY_MODE` | `fake` (default) / `cache` / `real` |
| `SATQUERY_TOOL_<tool>` | override the mode for one tool, e.g. `SATQUERY_TOOL_detector=real` |
| `SATQUERY_STUB_JOB` | `1` (default until BE-11) = `run_job` returns the stub stories; `0` = real pipeline |
| `SATQUERY_STUB_DELAY` | seconds to sleep between stub steps (shows the progress timeline) |
| `SATQUERY_FAKE_VARIANT` | e.g. `conflict`: fakes/stub return the "language model disagrees" version |
| `SATQUERY_SCENE` | when a real run saves to the cache, record its keys under this demo-scene name |
| `SATQUERY_CACHE_DIR`, `SATQUERY_FIXTURES_DIR`, `SATQUERY_ROOT` | relocate the cache / fixtures / repo root (Kaggle) |

## Try the stub

```python
from pathlib import Path
from satquery.agent.job import run_job
r = run_job("Has anything been built on the southern lake shore since last year?",
            [Path("fixtures/scenes/pair_T1.tif"), Path("fixtures/scenes/pair_T2.tif")], Path("/tmp/runs/demo"))
print(r.fusion.answer, r.fusion.confidence_label)
```
