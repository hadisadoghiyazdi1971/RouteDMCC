# Experiment Protocol

This note is derived from the supplied experiment result archive and the current manuscript. It does not add unreported experimental settings.

## Common settings in the main experiments

The reference JSON records, unless overridden by an individual experiment:

- nodes: `N = 30`;
- parameter dimension: `M = 2`;
- iterations: `2000`;
- Monte Carlo trials: `20`;
- step size: `mu = 0.05`;
- nominal noise standard deviation: `0.05`;
- MCC bandwidth: `sigma = 1.0`;
- routing threshold parameter: `kappa = 0.2`;
- EWMA forgetting factor: `gate_lambda = 0.95`;
- context scale: `gate_eta = 1.0`;
- context gain: `gate_context_gain = 0.5`;
- gate decision threshold: `1.0`.

Exact values for each experiment are stored in `configs/reference/*.json` and should take precedence over this summary.

## Randomness

Each main experiment stores a base `seed`. The original experiment program runs 20 trials using a deterministic trial-seed schedule derived from that base seed. The exact simulator source should be retained in `src/` so this schedule remains auditable rather than paraphrased.

## Aggregation

The supplied result archive stores mean and standard-deviation summaries and averaged trajectories. The manuscript states that reported results are averaged over 20 independent Monte Carlo trials and that `±` denotes one standard deviation across trials.

## Non-stationary contamination

The manuscript and experiment source use a three-regime impulse-probability schedule:

- iterations 1–600: `p_H = 0.05`;
- iterations 601–1400: `p_H = 0.30`;
- iterations 1401–2000: `p_H = 0.05`.

The frozen `nonstationary` config JSON does not store the schedule because the master result serializer omitted `impulse_prob_schedule`; the schedule is therefore documented here and should remain visible in the exact research source file.

## Interpretation discipline

The experiments test whether conditional robust processing can preserve the behavior of a uniformly robust branch while reducing robust-branch activation. They do not establish universal superiority across robust estimators, contamination distributions, or hardware implementations.
