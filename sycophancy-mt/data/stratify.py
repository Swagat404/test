"""Stratified sample: balanced across 3x3 (gold quality x ambiguity) tertiles per pair.

Why 3x3 instead of 3:
  Spearman(gold, ambiguity) on the en-hi WMT QE 2023 data is -0.31 — moderately
  negative. If we stratify only on ambiguity, our high-ambiguity tertile will be
  systematically biased toward low-quality translations. That lets a reviewer ask
  whether H3 (judges shift more on ambiguous segments) is really about ambiguity or
  just about translation badness. Cross-stratifying on a 3x3 grid forces the two
  axes to be roughly independent in the experimental sample, killing the confound
  by construction.

Cell layout per language pair (n_per_pair = 600):
                                ambiguity
                          low      mid     high
                       ┌───────┬────────┬───────┐
              low      │  ~67  │  ~67   │  ~67  │
       gold   mid      │  ~67  │  ~67   │  ~67  │
              high     │  ~67  │  ~67   │  ~67  │
                       └───────┴────────┴───────┘

If a cell is undersized (sparse cells along the rho=-0.31 diagonal), we take all
available and report the imbalance — better than silently dropping the cell.

seed = 42 throughout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TERTILE_LABELS = ["low", "mid", "high"]


def _tertile(series: pd.Series, labels=TERTILE_LABELS) -> pd.Series:
    """Robust tertile: qcut with duplicates='drop', fall back to equal-width cut."""
    try:
        return pd.qcut(series, q=3, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.cut(series, bins=3, labels=labels, include_lowest=True).astype(str)


def stratify(df: pd.DataFrame, n_per_pair: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for pair, group in df.groupby("lang_pair"):
        group = group.dropna(subset=["ambiguity", "gold"]).copy()
        group["ambiguity_tertile"] = _tertile(group["ambiguity"])
        group["gold_tertile"] = _tertile(group["gold"])

        n_per_cell = max(1, n_per_pair // 9)
        for g_tert in TERTILE_LABELS:
            for a_tert in TERTILE_LABELS:
                cell = group[
                    (group["gold_tertile"] == g_tert)
                    & (group["ambiguity_tertile"] == a_tert)
                ]
                if len(cell) == 0:
                    print(f"WARN: pair={pair} gold={g_tert} ambig={a_tert}: 0 segments (cell skipped)")
                    continue
                take = min(n_per_cell, len(cell))
                if take < n_per_cell:
                    print(f"NOTE: pair={pair} gold={g_tert} ambig={a_tert}: "
                          f"only {len(cell)} avail, wanted {n_per_cell}")
                idx = rng.choice(cell.index, size=take, replace=False)
                out.append(group.loc[idx])

    sample = pd.concat(out, ignore_index=True)
    if "seg_id" not in sample.columns:
        sample["seg_id"] = sample.index.astype(str)
    sample["seg_id"] = sample["seg_id"].astype(str)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/segments.parquet")
    ap.add_argument("--out", default="data/sample.parquet")
    ap.add_argument("--n_per_pair", type=int, default=600,
                    help="target sample size per language pair (~9 cells of n/9)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    sample = stratify(df, args.n_per_pair, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out, index=False)

    print(f"\nsampled {len(sample):,} segments -> {out}")
    print("\nCells per (lang_pair, gold_tertile, ambiguity_tertile):")
    print(sample.groupby(["lang_pair", "gold_tertile", "ambiguity_tertile"]).size().unstack(fill_value=0))

    rho = sample[["gold", "ambiguity"]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman(gold, ambiguity) in SAMPLE = {rho:+.3f}  "
          f"(was -0.310 in full data; closer to 0 = confound resolved)")


if __name__ == "__main__":
    main()
