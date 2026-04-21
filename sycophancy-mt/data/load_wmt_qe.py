"""Load WMT QE 2023 sentence-level DA TSVs (the official 2024 inputs reuse 2023 train/dev).

The 2024 task page explicitly says: "no training and validation datasets will be released
for Task 1; participants should use the previous year's data." The 2024 test files only
contain aggregated z-mean per segment. The per-rater data we need (to compute per-segment
ambiguity = SD across raters) lives in the 2023 train/dev TSVs.

Source repo: https://github.com/WMT-QE-Task/wmt-qe-2023-data
Pairs with per-rater DA: en-hi, en-ta, en-gu, en-te, en-mr (Indic only)

Input TSV columns (verified 2026-04-19 against actual files):
    index | original | translation | scores | mean | z_scores | z_mean

Where `scores` is a stringified Python list like "[90, 90, 79, 81]" and `z_scores`
is the per-rater z-standardized version. Standardization is already done by WMT; we
just compute SD across raters as our ambiguity moderator.

Output `segments.parquet` columns:
    seg_id, lang_pair, src_lang, tgt_lang, src, mt,
    raw_scores (list[int]), z_scores (list[float]),
    gold (mean of z_scores, == z_mean), ambiguity (SD of z_scores), n_raters
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT_DEFAULT = Path("data/raw/wmt-qe-2023")

PAIR_TO_LANGS: dict[str, tuple[str, str]] = {
    "en-hi": ("en", "hi"),
    "en-ta": ("en", "ta"),
    "en-gu": ("en", "gu"),
    "en-te": ("en", "te"),
    "en-mr": ("en", "mr"),
}


def _parse_list_cell(cell: str) -> list[float]:
    """Parse a stringified list like '[90, 90, 79, 81]' robustly."""
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    try:
        parsed = ast.literal_eval(s)
        return list(parsed) if not isinstance(parsed, list) else parsed
    except (ValueError, SyntaxError):
        s_clean = s.strip("[]")
        if not s_clean:
            return []
        return [float(x.strip()) for x in s_clean.split(",") if x.strip()]


def load_pair(pair: str, data_root: Path = DATA_ROOT_DEFAULT) -> pd.DataFrame:
    """Load one language pair: concatenate train + dev into a single segment-level table.

    Returns a DataFrame with one row per (source, translation) segment.
    """
    if pair not in PAIR_TO_LANGS:
        raise ValueError(f"unknown pair {pair!r}; expected one of {list(PAIR_TO_LANGS)}")

    pair_short = pair.replace("-", "")
    pair_dir = data_root / pair
    files = [
        pair_dir / f"train.{pair_short}.df.short.tsv",
        pair_dir / f"dev.{pair_short}.df.short.tsv",
    ]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing TSV(s) for {pair}: {missing}. "
            f"Expected layout: {pair_dir}/{{train,dev}}.{pair_short}.df.short.tsv"
        )

    frames = []
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        df["__source_file"] = fp.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df["raw_scores"] = df["scores"].apply(_parse_list_cell)
    df["z_scores_list"] = df["z_scores"].apply(_parse_list_cell)
    df["n_raters"] = df["raw_scores"].apply(len)
    df["ambiguity"] = df["z_scores_list"].apply(
        lambda xs: float(np.std(xs, ddof=0)) if len(xs) >= 2 else np.nan
    )
    df["gold"] = df["z_mean"].astype(float)

    src_lang, tgt_lang = PAIR_TO_LANGS[pair]
    out = pd.DataFrame({
        "seg_id": df.apply(
            lambda r: f"{pair}:{r['__source_file']}:{int(r['index'])}", axis=1
        ),
        "lang_pair": pair,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src": df["original"].astype(str),
        "mt": df["translation"].astype(str),
        "raw_scores": df["raw_scores"],
        "z_scores": df["z_scores_list"],
        "gold": df["gold"],
        "ambiguity": df["ambiguity"],
        "n_raters": df["n_raters"].astype(int),
    })
    out = out[(out["n_raters"] >= 3) & out["ambiguity"].notna()].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["en-hi"],
                    help="WMT QE 2023 Indic DA pairs (en-hi, en-ta, en-gu, en-te, en-mr)")
    ap.add_argument("--data_root", default=str(DATA_ROOT_DEFAULT),
                    help="root containing one folder per pair with train.*.tsv and dev.*.tsv")
    ap.add_argument("--out", default="data/segments.parquet")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    frames = [load_pair(p, data_root) for p in args.pairs]
    df = pd.concat(frames, ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"wrote {len(df):,} segments -> {out}")
    print()
    print("Per-pair counts:")
    print(df["lang_pair"].value_counts().to_string())
    print()
    print("Ambiguity distribution per pair (SD of z_scores across raters):")
    summary = df.groupby("lang_pair")["ambiguity"].describe()[
        ["count", "min", "25%", "50%", "75%", "max"]
    ]
    print(summary.to_string())
    print()
    print("Gold distribution per pair (mean z-score):")
    print(df.groupby("lang_pair")["gold"].describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    print()
    rho = df[["gold", "ambiguity"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(gold, ambiguity) = {rho:+.3f}  "
          f"({'WATCH: confound risk' if abs(rho) > 0.3 else 'OK: low confound'})")


if __name__ == "__main__":
    main()
