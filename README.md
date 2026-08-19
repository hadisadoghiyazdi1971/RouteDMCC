# RouteDMCC

**Sample-Level Computational Routing for Efficient Robust Diffusion Adaptation over Heterogeneous Networks**

RouteDMCC is the reproducibility companion for the manuscript by **Soheila Ashkezari-Toussi** and **Hadi Sadoghi-Yazdi**. The work studies a context-aware routing rule that selects, at each node and sample, between a low-cost LMS update and a robust bounded-influence update. The main instantiation uses diffusion maximum correntropy criterion (DMCC) on the hard route.

## Why this repository exists

The manuscript makes two empirical claims that should be easy to audit:

1. gated robust processing can remain close to uniform robust processing in steady-state MSD in the evaluated impulsive-noise regimes; and
2. the robust branch can be activated for only a fraction of observations, reducing the paper's **active-cost proxy**, while wall-clock savings are more modest.

This repository is organized so that an editor or reviewer can move directly from a paper table/figure to the corresponding configuration and reference output.

## Reviewer quick path

1. Read [`docs/PAPER_TO_ARTIFACT_MAP.md`](docs/PAPER_TO_ARTIFACT_MAP.md).
2. Inspect exact configuration snapshots in [`configs/reference/`](configs/reference/).
3. Inspect the unchanged reference JSON/PNG artifacts in [`results/reference/`](results/reference/).
4. Run `python tools/verify_reference_hashes.py` to check the packaged reference-artifact hashes.
5. Run `python tools/export_reference_summaries.py` to regenerate the human-readable CSV summaries from the reference JSON.
6. See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for seeds, Monte Carlo aggregation, contamination regimes, and interpretation limits.

## Repository status

| Component | Status |
|---|---|
| Reference result JSON files | Included |
| Reference figures | Included |
| Exact config snapshots recovered from reference JSON | Included |
| Paper-to-artifact map | Included |
| Integrity/hash verification | Included |
| Human-readable summary export | Included |
| Executable research simulator | **Add the original `routeDMCC.py` to `src/` before archival release** |
| Frozen runtime versions | **To be recorded from the original execution environment** |

The executable experiment file was identified in the authors' research archive, but it was not embedded in the submission/result ZIP used to assemble this GitHub package. It is intentionally **not reimplemented here**, because a rewritten implementation could silently change the experiments. The public archival release should contain the exact original experiment file.

## Experimental data provenance

No external benchmark dataset is required for the reported experiments. Observations are synthetically generated according to the stochastic models and parameters used by the experiment code. The packaged outputs therefore provide reference results rather than a third-party dataset. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Main experimental coverage

- stationary Gaussian implementation validation;
- heterogeneous node-noise validation;
- impulsive contamination at multiple impulse probabilities;
- uniform and gated DMCC;
- Sign-Error, Huber, and Gated Huber controls;
- strong-impulse stress testing;
- non-stationary impulse-probability schedules;
- network-size sweep;
- `kappa/sigma` sensitivity;
- context-gain sensitivity;
- active-cost and wall-clock reporting.

## Key files

- `results/reference/dlms_experiment_results.json` — master result archive for the main experiments.
- `configs/reference/` — exact config blocks extracted from that archive.
- `results/reference/summaries/` — CSV views generated without rounding the stored values.
- `docs/PAPER_TO_ARTIFACT_MAP.md` — mapping from manuscript tables/figures to repository artifacts.
- `docs/COMPLEXITY_MODEL.md` — interpretation of the active-cost proxy.
- `docs/REVIEWER_GUIDE.md` — short audit path and claim boundaries.
- `REFERENCE_ARTIFACT_SHA256.txt` — hashes for the unchanged source artifacts copied into the repository.

## Reproduction

The current package supports **artifact verification** immediately. Full simulation reruns require placing the exact original experiment program in `src/routeDMCC.py`.

After the source file is present:

```bash
python -m pip install -r environment/requirements-minimal.txt
python src/routeDMCC.py
```

The original program writes its outputs to `code_results/`. Keep those regenerated outputs separate from `results/reference/`; the latter is the frozen comparison target.

## Integrity

```bash
python tools/verify_reference_hashes.py
```

A successful verification means the frozen JSON and PNG reference artifacts match the files used to build this repository snapshot. It does **not** by itself prove independent reproduction of the experiments.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Once the manuscript has a DOI, update the preferred citation and archive the submission snapshot (for example, as a GitHub release and a DOI-bearing archival record).

## License

This repository is released under the MIT License. Authors should confirm that this license is appropriate for any additional third-party code added later.
