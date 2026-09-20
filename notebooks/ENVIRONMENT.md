# ML-00: Environment record (P1)

Recorded 2026-09-20. Versions per master F3.9; conventions per F3.2.

## Verified

| | Local | Kaggle |
|---|---|---|
| OS / Python | WSL2 Ubuntu 26.04, **Python 3.12.14** via `uv` | **Python 3.12.13** |
| GPU | none | 2x Tesla T4, **14.56 GiB each**, sm_75 |
| torch | n/a | 2.10.0+cu128 (driver CUDA 13.0) |
| rasterio / GDAL | 1.5.1 / 3.12.4 | 1.5.0 |
| transformers | n/a | **5.0.0** |
| numpy | 2.5.3 | **2.0.2** |
| bf16 | n/a | **emulated only — use fp16** |

## Three things the team needs to know

**1. Kaggle runs Python 3.12, not 3.11.** Verified from `Kaggle/docker-python`
`Dockerfile.tmpl` (`PACKAGE_PATH=/usr/local/lib/python3.12/dist-packages`).
Master F3.2 says "Python 3.11 for everyone", but the machine where all real
inference happens is 3.12. Both satisfy `requires-python = ">=3.10,<3.13"`, so
no `pyproject.toml` change is needed — but the team should standardise on
**3.12** so laptops and Kaggle resolve the same wheels.

**2. Windows Smart App Control blocks rasterio.** On Windows 11 with Smart App
Control enforced (`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` ->
`VerifiedAndReputablePolicyState = 1`), `import rasterio` fails with
`DLL load failed while importing _base: An Application Control policy has
blocked this file`. CodeIntegrity events 3033/3077/3118 name
`rasterio.libs\libpq-<hash>.dll` as unsigned. The filename carries a per-build
hash so it will never earn Microsoft cloud reputation. `shapely` and `pyproj`
are unaffected — only rasterio's bundled GDAL DLLs.

*Workaround used:* WSL2 + Python 3.12 via `uv`. Linux `.so` files aren't
governed by Smart App Control. Turning SAC off also works but is irreversible
without resetting Windows, so it wasn't done.

**3. numpy differs between laptop (2.5.3) and Kaggle (2.0.2).** Pin it on Day 0
with `pip freeze` as master F3.9 suggests. When running
`pip install -e ".[geo,models]"` on Kaggle, re-check `numpy.__version__`
afterwards — Kaggle's preinstalled torch and timm are built against 2.0.x and a
silent bump breaks them.

## Setup

**Local (WSL2, recommended):**

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    cd ~/SatQuery-AI-Developer-Divas
    uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install -e ".[geo,server,dev]"     # once P2's pyproject.toml exists

Keep the repo on the Linux filesystem, not `/mnt/c/...` — cross-mount I/O is
roughly an order of magnitude slower and OneDrive will try to sync `.venv`.

Note: handover ML-00 says `".[geo,dev]"`, master F3.9 says `".[geo,server,dev]"`
plus `playwright install chromium`. Master wins and `[server]` is a superset;
you need it for the daily `pytest tests/e2e` run.

**Kaggle:** Accelerator = GPU T4 x2, Internet = On, then
`pip install -e ".[geo,models]"`. Do not build a venv — use the system
interpreter.

## Verification

    pytest tests/p1/test_env.py -v              # local: 13 passed, 8 skipped
    SATQUERY_EXPECT_GPU=1 pytest tests/p1/test_env.py -v -s   # Kaggle

The 8 local skips are the GPU half plus `satquery.contracts`, which is blocked
on P2's Day 0 skeleton.

## Blocked on P2

`pyproject.toml` (extras `geo`, `models`, `server`, `dev`), `satquery/contracts/`,
`satquery/common/` (`toolbase`, `geo`, `cache`), `tests/contract/`, and
`fixtures/make_fixtures.py`. ML-02 onward cannot start without them.
