"""Corrected state codes as an overlay (normalized/state_fix_<file>.parquet: entity_id, state).

The parser fix in normalize.py (Washington DC vs Washington state, "Washington, Utah", "Ohio, NY") changes the state of
~0.3% of rows. Blocking barely depends on the state, so instead of re-running normalization, blocking and the dense
search, the ranker reads this overlay. `python normalize.py` produces the same values from scratch.

  python fix_states.py            -> overlay for train and test
"""
import os
import time
import multiprocessing as mp

import polars as pl

import normalize as N

NORM = N.OUT


def _chunk(args):
    addrs, ctry = args
    return [N.parse_state(a, c) for a, c in zip(addrs, ctry)]


def build(prefix: str, n: int, pool):
    src = NORM / f"{prefix}source{n}.parquet"
    d = pl.read_parquet(src, columns=["entity_id", "addr_raw", "country"])
    step = 100_000
    chunks = [(d["addr_raw"].slice(i, step).to_list(), d["country"].slice(i, step).to_list()) for i in range(0, d.height, step)]
    st = [x for part in pool.map(_chunk, chunks) for x in part]
    out = NORM / f"state_fix_{prefix}source{n}.parquet"
    d.select("entity_id").with_columns(pl.Series("state", st)).write_parquet(out)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("ER_WORKERS", min(8, os.cpu_count() or 2)))) as pool:
        for prefix in ("", "test_"):
            for n in (1, 2, 3):
                print(build(prefix, n, pool), f"{time.time() - t0:.0f}s", flush=True)
