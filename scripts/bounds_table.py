#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Summarize `run_scripted_sweep.py --plan bounds` runs: one row per renderer timing.

For each timing: the bound [T_lead + T_future, T_lead + T_future + T_present], and for the floor
set and the ceiling set the measured RLC and ILC (wall clock) as mean [min, max] in ms, against
the prediction. Also the largest error to the per-turn prediction. Standard library only.

Usage:
    python scripts/bounds_table.py framewise.jsonl blockwise.jsonl avtr1.jsonl
"""

import argparse
import json
from pathlib import Path
from statistics import mean


def _fmt(values: list[float]) -> str:
    return f"{mean(values):6.1f} [{min(values):5.1f}, {max(values):5.1f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("jsonl", type=Path, nargs="+")
    args = p.parse_args()

    for path in args.jsonl:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        t = rows[0]["timing"]
        floor, present = t["lead"] + t["future"], t["present"]
        print(
            f"\n== {t['name']}: T_present {present}, T_future {t['future']}, T_lead {t['lead']}, "
            f"TTFF {t['ttff']}, MTBF {t['mtbf']} (ms); bound [{floor}, {floor + present}]"
        )
        errors = []
        for label in ("floor", "ceiling"):
            subset = [r for r in rows if r["config"]["label"] == label]
            for kind in ("rlc", "ilc"):
                done = [r for r in subset if r.get(kind)]
                wall = [r[kind]["t_total_wall"] * 1000 for r in done]
                pred = sorted({round(r["predicted"][kind] * 1000, 1) for r in done})
                errors += [w - r["predicted"][kind] * 1000 for w, r in zip(wall, done, strict=True)]
                print(
                    f"  {label:7s} {kind.upper()}: n={len(wall):2d}  measured {_fmt(wall)}  predicted {pred}"
                )
        inside = sum(
            1
            for r in rows
            for kind in ("rlc", "ilc")
            if r.get(kind) and floor <= r[kind]["t_total_wall"] * 1000 <= floor + present
        )
        print(
            f"  wall - prediction: {min(errors):+.2f} .. {max(errors):+.2f} ms; "
            f"inside the bound: {inside} of {len(errors)}"
        )


if __name__ == "__main__":
    main()
