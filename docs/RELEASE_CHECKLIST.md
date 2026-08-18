# Archival Release Checklist

Before creating `v1.0.0-submission`:

- [ ] Place the exact original `dlms_experiment3_GLM2_3.py` in `src/`.
- [ ] Run it in the original or documented compatible environment.
- [ ] Record Python, NumPy, Matplotlib, OS, CPU, and thread/backend details relevant to wall-clock measurements.
- [ ] Keep regenerated outputs outside `results/reference/` until compared.
- [ ] Compare regenerated JSON summaries with the frozen reference artifacts.
- [ ] Run `python tools/verify_reference_hashes.py`.
- [ ] Run `python tools/export_reference_summaries.py` and confirm there is no unexpected diff.
- [ ] Update `CITATION.cff` with the repository URL and, later, manuscript DOI.
- [ ] Create the GitHub release/tag `v1.0.0-submission`.
- [ ] Archive that release with a persistent DOI if desired.

Do not label the repository as a full independent reproduction until the exact experiment source and runtime description are present.
