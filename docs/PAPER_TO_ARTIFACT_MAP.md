# Paper-to-Artifact Map

The mapping below follows the current 10-page manuscript structure.

| Manuscript item | Scientific role | Reference artifact | Config / source |
|---|---|---|---|
| Table I — implementation validation | Gaussian baseline + heterogeneous node noise | `dlms_experiment_results.json`: `base_gaussian_dlms`, `noisy_node_weighting` | `configs/reference/base_gaussian_dlms.json`, `configs/reference/noisy_node_weighting.json` |
| Table II — impulsive-noise results | `p_H=0.05` and `p_H=0.10` gated-vs-uniform robust adaptation | `dlms_experiment_results.json`: `impulsive_asymmetric_dlms`, `impulsive_p10_dlms` | corresponding config JSON files |
| Table III — robust baselines | Sign-Error, Huber, Uniform DMCC, Gated DMCC, Gated Huber + wall clock | `dlms_experiment_results.json`: `robust_baselines`; `wall_clock_baselines.png` | `configs/reference/robust_baselines.json` |
| Table IV — stress tests | strong impulses + non-stationary contamination | `dlms_experiment_results.json`: `strong_impulse`, `nonstationary` | corresponding config JSON files |
| Fig. non-stationary curves | adaptation and gate response under `0.05 -> 0.30 -> 0.05` | `results/reference/figures/nonstationary_curves.png` | `configs/reference/nonstationary.json` |
| Impulse-probability sweep table/plot | how cost and MSD change with contamination frequency | `impulse_prob_sweep_summary.json`, `impulse_prob_sweep.png`, `impulse_rate_comparison.png` | sweep summary is frozen reference output |
| Network-size sweep table/plot | scale with `N` while keeping `M=2` fixed | `network_size_sweep_summary.json`, `network_size_sweep.png` | sweep summary is frozen reference output |
| `kappa/sigma` sensitivity table/plot | routing/robust-scale sensitivity | `kappa_sigma_sweep_summary.json`, `kappa_sigma_sweep.png`, `kappa_sigma_sensitivity.png` | sweep summary is frozen reference output |
| Decision-boundary figure | definition-level visualization of the multiplicative gate | `decision_boundary.png` | gate definition in research source/manuscript |
| Context-gain table | sensitivity to `lambda_c` | `results/paper_reported/lambda_context_gain_table.csv` | values transcribed from manuscript; raw summary not present in supplied result archive |
| Context-gain non-stationary figure | effect of `lambda_c` on MSD and gate activity | `lambda_c_nonstationary_curves.png` | reference figure supplied with result archive |

## Important distinction

Files in `results/reference/` are frozen source artifacts copied from the authors' result archive. Files in `results/paper_reported/` are explicitly labeled manuscript transcriptions when the machine-generated summary file was not present in that archive.
