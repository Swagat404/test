"""Async runner: iterate (segment x judge x prompt_family x condition).

Resumable via append-only JSONL: at startup, scan results.jsonl for already-
computed (seg_id, judge, prompt_family) keys and skip them.

Usage:
  python -m src.runner --config configs/full.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from .judges import BaseJudge, get_judge
from .prompts import (
    NEUTRAL_CONTROL,
    build_challenge,
    gold_inconsistent_direction,
    render_baseline,
    render_simultaneous,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_done_keys(results_path: Path) -> set[tuple]:
    done = set()
    if not results_path.exists():
        return done
    with results_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row["seg_id"], row["judge"], row["prompt_family"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


async def run_one_segment(
    seg: dict,
    judge: BaseJudge,
    prompt_family: str,
    challenge_kinds: list[str],
) -> dict:
    """Execute baseline + 3 challenges + neutral + simultaneous for one segment.

    Returns a single flat dict that is written as one JSONL row.
    """
    src_lang = seg["src_lang"]
    tgt_lang = seg["tgt_lang"]
    src_seg = seg["src"]
    tgt_seg = seg["mt"]
    gold = float(seg["gold"])

    sys_prompt, user_prompt = render_baseline(prompt_family, src_lang, tgt_lang, src_seg, tgt_seg)
    base = await judge.baseline_score(sys_prompt, user_prompt)
    s_0 = base.score

    direction = gold_inconsistent_direction(s_0 if s_0 is not None else 50.0, gold)

    out: dict = {
        "seg_id": seg["seg_id"],
        "lang_pair": seg["lang_pair"],
        "judge": judge.name,
        "prompt_family": prompt_family,
        "gold": gold,
        "ambiguity": float(seg["ambiguity"]),
        "ambiguity_tertile": seg.get("ambiguity_tertile"),
        "direction": direction,
        "s_0": s_0,
        "s_0_raw": base.raw,
        "s_0_error": base.error,
    }

    representative_critique = None

    challenge_tasks = []
    for kind in challenge_kinds:
        ch = build_challenge(kind, direction)
        challenge_tasks.append((kind, ch.text, judge.followup_score(base.history, ch.text)))
        if representative_critique is None:
            representative_critique = ch.text

    neutral_task = judge.followup_score(base.history, NEUTRAL_CONTROL)

    sim_sys, sim_user = render_simultaneous(
        src_lang, tgt_lang, src_seg, tgt_seg,
        critique=representative_critique or "",
    )
    sim_task = judge.baseline_score(sim_sys, sim_user)

    challenge_results = await asyncio.gather(*[t[2] for t in challenge_tasks])
    neutral_result, sim_result = await asyncio.gather(neutral_task, sim_task)

    for (kind, text, _), result in zip(challenge_tasks, challenge_results):
        out[f"s_c_{kind}"] = result.score
        out[f"s_c_{kind}_raw"] = result.raw
        out[f"s_c_{kind}_error"] = result.error

    out["s_neutral"] = neutral_result.score
    out["s_neutral_raw"] = neutral_result.raw
    out["s_neutral_error"] = neutral_result.error

    out["s_simultaneous"] = sim_result.score
    out["s_simultaneous_raw"] = sim_result.raw
    out["s_simultaneous_error"] = sim_result.error

    return out


async def main_async(cfg: dict) -> None:
    load_dotenv()

    sample_path = Path(cfg["sample_path"])
    results_path = Path(cfg["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(sample_path)
    print(f"loaded {len(df)} segments from {sample_path}")

    done = load_done_keys(results_path)
    print(f"resuming: {len(done)} (seg, judge, prompt) combos already complete")

    judges: dict[str, BaseJudge] = {name: get_judge(name) for name in cfg["judges"]}
    prompt_families: list[str] = cfg["prompt_families"]
    challenge_kinds: list[str] = cfg["challenges"]

    work = []
    for _, seg_row in df.iterrows():
        seg = seg_row.to_dict()
        for jname, judge in judges.items():
            for fam in prompt_families:
                if (seg["seg_id"], jname, fam) in done:
                    continue
                work.append((seg, judge, fam))

    print(f"{len(work)} (seg x judge x prompt_family) tasks queued")

    checkpoint_every = int(cfg.get("checkpoint_every", 50))
    written = 0
    t0 = time.time()

    with results_path.open("a") as fout:
        async def runner(item):
            nonlocal written
            seg, judge, fam = item
            try:
                row = await run_one_segment(seg, judge, fam, challenge_kinds)
            except Exception as e:
                row = {
                    "seg_id": seg["seg_id"], "lang_pair": seg["lang_pair"],
                    "judge": judge.name, "prompt_family": fam,
                    "fatal_error": f"{type(e).__name__}: {e}",
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % checkpoint_every == 0:
                fout.flush()
                os.fsync(fout.fileno())
            return row

        await tqdm_asyncio.gather(*[runner(item) for item in work])
        fout.flush()
        os.fsync(fout.fileno())

    print(f"done in {time.time() - t0:.1f}s, wrote {written} rows -> {results_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    sys.exit(main() or 0)
