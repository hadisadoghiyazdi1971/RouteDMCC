# Reviewer Guide

This repository is intended to make the manuscript's empirical evidence easy to audit without requiring a reviewer to reverse-engineer file names.

## 90-second audit

1. Open `PAPER_TO_ARTIFACT_MAP.md` and locate the paper item of interest.
2. Open the linked config snapshot under `configs/reference/`.
3. Inspect the matching entry in `results/reference/dlms_experiment_results.json` or the relevant sweep JSON.
4. Compare the reference figure under `results/reference/figures/` when applicable.
5. Run the hash checker if artifact integrity matters for the review.

## What the repository supports

The supplied artifacts support direct auditing of reported configurations, aggregate metrics, averaged trajectories, sensitivity sweeps, gate activation rates, active-cost values, and reference plots.

## What should not be inferred

- The active-cost metric is a branch-activation cost proxy, not a hardware-normalized runtime model.
- Wall-clock measurements are implementation-dependent descriptive measurements.
- Similarity between Gated DMCC and Uniform DMCC is supported only in the evaluated regimes; the paper does not claim uniform superiority.
- Huber obtains lower MSD in some evaluated regimes; that negative/mixed evidence is intentionally retained.
- The decision-boundary figure visualizes the defined gate. It is not an empirical phase transition.

## Full rerun status

The exact original simulator should be placed at `src/dlms_experiment3_GLM2_3.py` before the archival release. The current package deliberately avoids substituting a newly written implementation for the code that generated the reported results.
