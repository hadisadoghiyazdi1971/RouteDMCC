#!/usr/bin/env python3
"""Export human-readable CSV summaries from frozen RouteDMCC JSON artifacts.
No experiment is rerun and no stored numeric value is modified or rounded.
"""
from pathlib import Path
import json, csv
ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference"
OUT = REF / "summaries"
OUT.mkdir(parents=True, exist_ok=True)
master = json.loads((REF / "dlms_experiment_results.json").read_text(encoding="utf-8"))
fields = ["experiment","algorithm","msd_mean","msd_std","emse_mean","emse_std","cost_mean","cost_std","robust_rate_mean","robust_rate_std","wall_clock_sec_mean","wall_clock_sec_std"]
with (OUT / "main_experiment_summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for exp in master:
        for alg, vals in exp["summary"].items():
            if alg == "empirical_hard_rate" or not isinstance(vals, dict):
                continue
            row = {"experiment": exp["label"], "algorithm": alg}
            for k in fields[2:]: row[k] = vals.get(k, "")
            w.writerow(row)
for stem in ["impulse_prob_sweep_summary", "kappa_sigma_sweep_summary", "network_size_sweep_summary"]:
    p = REF / f"{stem}.json"
    if not p.exists(): continue
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows: continue
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"Wrote reference CSV views to {OUT}")
