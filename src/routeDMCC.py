"""
Distributed Adaptive Filtering Experiments
==========================================
Final version with Multiplicative Context-Aware Gating.

KEY INNOVATION (Multiplicative Context Gate):
-----------------------------------------------
The routing gate uses a context-aware hardness score that multiplicatively
modulates the instantaneous error:

    h_k(n) = (|e_k(n)| / κ) * [1 + λ_c * (1 - exp(-σ_e / η))]
    r_k(n) = 1{ h_k(n) > threshold }

where σ_e = sqrt(p̂_k(n)) and p̂_k(n) is an EWMA of squared error.

WHY THIS IS MATHEMATICALLY SUPERIOR TO ADDITIVE GATES:
1. If |e| = 0, h = 0 strictly. The gate NEVER fires for zero error, 
   regardless of context. This preserves computational sparsity perfectly.
2. Context (σ_e) acts as a sensitivity GAIN on the error, not an additive
   offset. It changes the effective threshold dynamically.
3. Irreducible to a classical M-estimator (like Huber). 
   Huber: ψ(e) — depends ONLY on e.
   Ours:  Ψ(e, σ_e) — depends on BOTH e AND error history.
"""

import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
try:
    fm.fontManager.addfont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
except Exception:
    pass
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================================
# Display names & Styles
# ==========================================================
DISPLAY_NAMES = {
    "noncoop_lms": "Non-cooperative LMS",
    "global_lms": "Global LMS",
    "cta_lms_CI": "CTA DLMS, C=I",
    "atc_lms_CI": "ATC DLMS, C=I",
    "atc_lms_CneqI": "ATC DLMS, C!=I",
    "atc_lms_rdv_CI": "ATC RDV weights",
    "uniform_dmcc_atc": "Uniform DMCC ATC",
    "gated_dmcc_atc": "Gated DMCC ATC",
    "gated_dmcc_cta": "Gated DMCC CTA",
    "gated_huber_atc": "Gated Huber ATC",
    "sign_error_atc": "Sign-Error ATC",
    "huber_atc": "Huber ATC",
}

ALGO_COLORS = {
    "noncoop_lms":        "#777777",
    "global_lms":         "#17becf",
    "cta_lms_CI":         "#bcbd22",
    "atc_lms_CI":         "#777777",
    "atc_lms_CneqI":      "#8c564b",
    "atc_lms_rdv_CI":     "#7f7f7f",
    "uniform_dmcc_atc":   "#d62728",
    "gated_dmcc_atc":     "#1f77b4",
    "gated_dmcc_cta":     "#aec7e8",
    "gated_huber_atc":    "#2ca02c",
    "sign_error_atc":     "#9467bd",
    "huber_atc":          "#ff7f0e",
}

ALGO_ZORDER = {
    "atc_lms_CI": 2, "cta_lms_CI": 2, "noncoop_lms": 2, "global_lms": 2,
    "sign_error_atc": 3, "huber_atc": 3, "uniform_dmcc_atc": 3,
    "gated_dmcc_atc": 5, "gated_dmcc_cta": 5, "gated_huber_atc": 5,
    "atc_lms_CneqI": 2, "atc_lms_rdv_CI": 2,
}

ALGO_LINESTYLE = {
    "noncoop_lms": "-", "global_lms": "-", "cta_lms_CI": "--",
    "atc_lms_CI": "-", "atc_lms_CneqI": "-.", "atc_lms_rdv_CI": ":",
    "uniform_dmcc_atc": "-", "gated_dmcc_atc": "--", "gated_dmcc_cta": "-.",
    "gated_huber_atc": "--", "sign_error_atc": "-", "huber_atc": "-",
}

ALGO_MARKER = {
    "noncoop_lms": "o", "global_lms": "*", "cta_lms_CI": "s",
    "atc_lms_CI": "o", "atc_lms_CneqI": "D", "atc_lms_rdv_CI": "P",
    "uniform_dmcc_atc": "s", "gated_dmcc_atc": "^", "gated_dmcc_cta": "v",
    "gated_huber_atc": "d", "sign_error_atc": "X", "huber_atc": "h",
}

def _get_color(name): return ALGO_COLORS.get(name, "#333333")
def _get_zorder(name): return ALGO_ZORDER.get(name, 3)
def _get_linestyle(name): return ALGO_LINESTYLE.get(name, "-")
def _get_marker(name): return ALGO_MARKER.get(name, "o")
def _markevery(n_points, target_markers=10): return max(1, int(round(n_points / max(1, target_markers))))

def _annotate_regimes(ax, schedule, y_pos_frac=0.85):
    for (start, end, p) in schedule:
        ax.axvline(end, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        mid = (start + end) / 2
        ymin, ymax = ax.get_ylim()
        if ymin > 0:
            y_text = 10 ** (np.log10(ymin) + y_pos_frac * (np.log10(ymax) - np.log10(ymin)))
        else:
            y_text = ymax * y_pos_frac
        ax.text(mid, y_text, fr"$p_H={p:g}$", ha="center", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", pad=2))


# ==========================================================
# Topology and weighting
# ==========================================================
def connected_topology(n_nodes, radius=0.34, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(200):
        xy = rng.uniform(size=(n_nodes, 2))
        dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        adj = (dist <= radius).astype(float)
        np.fill_diagonal(adj, 1.0)
        seen = {0}
        stack = [0]
        while stack:
            k = stack.pop()
            for l in np.where(adj[:, k] > 0)[0]:
                if int(l) not in seen:
                    seen.add(int(l))
                    stack.append(int(l))
        if len(seen) == n_nodes:
            return adj, xy
    adj = np.ones((n_nodes, n_nodes), dtype=float)
    return adj, rng.uniform(size=(n_nodes, 2))

def metropolis_weights(adj):
    n_nodes = adj.shape[0]
    weights = np.zeros_like(adj, dtype=float)
    degrees = adj.sum(axis=0)
    for k in range(n_nodes):
        neighbors = np.where(adj[:, k] > 0)[0]
        for l in neighbors:
            if l == k: continue
            weights[l, k] = 1.0 / max(degrees[k], degrees[l])
        weights[k, k] = 1.0 - weights[:, k].sum()
    return weights

def relative_degree_weights(adj):
    n_nodes = adj.shape[0]
    weights = np.zeros_like(adj, dtype=float)
    degrees = adj.sum(axis=0)
    inv_deg = 1.0 / degrees
    for k in range(n_nodes):
        neighbors = np.where(adj[:, k] > 0)[0]
        weights[neighbors, k] = inv_deg[neighbors] / inv_deg[neighbors].sum()
    return weights

def relative_degree_variance_weights(adj, noise_var):
    n_nodes = adj.shape[0]
    weights = np.zeros_like(adj, dtype=float)
    degrees = adj.sum(axis=0)
    score = 1.0 / (degrees * noise_var)
    for k in range(n_nodes):
        neighbors = np.where(adj[:, k] > 0)[0]
        weights[neighbors, k] = score[neighbors] / score[neighbors].sum()
    return weights

def make_regressor_profile(n_nodes, m_dim):
    covs = []
    for k in range(n_nodes):
        scale = 0.45 + 1.1 * (k / max(1, n_nodes - 1))
        covs.append(np.diag(np.linspace(scale, scale * 1.6, m_dim)))
    return np.asarray(covs)

def sample_data(rng, w0, reg_covs, noise_std, impulse_prob=0.0, impulse_std=0.0):
    n_nodes, m_dim, _ = reg_covs.shape
    u = np.empty((n_nodes, m_dim))
    for k in range(n_nodes):
        u[k] = rng.multivariate_normal(np.zeros(m_dim), reg_covs[k])
    hard_mask = rng.uniform(size=n_nodes) < impulse_prob
    noise = np.where(
        hard_mask,
        rng.normal(0, impulse_std, n_nodes),
        rng.normal(0, noise_std, n_nodes),
    )
    d = u @ w0 + noise
    return u, d, hard_mask


# ==========================================================
# Influence function (Expert) and Gate logic (Routing)
# ==========================================================
def influence(error, mode, kappa, sigma=1.0, huber_delta=None, is_hard=False):
    """
    Returns (grad_err, relative_flop_cost).
    The gate logic (is_hard) is strictly separated and evaluated externally.
    """
    if huber_delta is None: huber_delta = kappa
    
    if mode == "lms":
        return error, 1.0
    if mode == "sign":
        return np.sign(error), 2.0
    if mode == "huber":
        if abs(error) <= huber_delta: return error, 4.0
        return huber_delta * np.sign(error), 4.0
    if mode == "dmcc":
        weight = np.exp(-(error ** 2) / (2 * sigma ** 2))
        return error * weight, 10.0
        
    # Gated Experts
    if mode == "gated_dmcc":
        if is_hard:
            weight = np.exp(-(error ** 2) / (2 * sigma ** 2))
            return error * weight, 10.0
        return error, 1.0
    if mode == "gated_huber":
        if is_hard:
            if abs(error) <= huber_delta: return error, 4.0
            return huber_delta * np.sign(error), 4.0
        return error, 1.0
    raise ValueError(f"unknown influence mode: {mode}")


def adapt_from_state(
        u, d, state, mu, s_mat,
        mode, kappa, sigma,
        huber_delta=None,
        context_state=None,
        gate_lambda=0.95,          # EWMA forgetting factor
        gate_eta=None,             # Context scale
        gate_context_gain=0.5,     # λ_c
        gate_threshold=1.0):

    n_nodes, m_dim = u.shape
    psi = np.zeros_like(state)
    active_cost = 0.0
    robust_count = 0
    eval_count = 0

    if gate_eta is None:
        gate_eta = kappa

    if context_state is None:
        context_state = np.zeros(n_nodes)

    new_context_state = context_state.copy()

    for k in range(n_nodes):
        update = np.zeros(m_dim)
        for l in np.where(s_mat[:, k] > 0)[0]:
            err = float(d[l] - u[l] @ state[:, k])

            # -------------------------------------------------
            # Context update (EWMA of error power)
            # -------------------------------------------------
            new_context_state[l] = (
                gate_lambda * context_state[l]
                + (1.0 - gate_lambda) * err**2
            )

            sigma_e = np.sqrt(new_context_state[l] + 1e-12)

            # Saturating context in [0,1]
            context = 1.0 - np.exp(
                -sigma_e / (gate_eta + 1e-12)
            )

            # -------------------------------------------------
            # Routing decision (Multiplicative Gate)
            # -------------------------------------------------
            if "gated" in mode:

                # Context-modulated hardness score
                h = (abs(err) / kappa) * (
                        1.0 + gate_context_gain * context
                    )

                is_hard = (h > gate_threshold)

            else:
                # always use robust expert for non-gated robust methods
                is_hard = mode in ("dmcc", "huber", "sign")

            # -------------------------------------------------
            # Expert Execution
            # -------------------------------------------------
            grad_err, cost_val = influence(
                err,
                mode,
                kappa,
                sigma,
                huber_delta=huber_delta,
                is_hard=is_hard,
            )

            update += s_mat[l, k] * u[l] * grad_err

            active_cost += cost_val
            robust_count += int(is_hard)
            eval_count += 1

        psi[:, k] = state[:, k] + mu * update

    return (
        psi,
        active_cost / max(eval_count, 1),
        robust_count / max(eval_count, 1),
        new_context_state,
    )

def step_atc(u, d, w, mu, a_mat, s_mat, mode, kappa, sigma, huber_delta=None, **kwargs):
    psi, cost, robust_rate, ctx_state = adapt_from_state(
        u, d, w, mu, s_mat, mode, kappa, sigma, huber_delta, **kwargs
    )
    return psi @ a_mat, cost, robust_rate, ctx_state

def step_cta(u, d, w, mu, a_mat, s_mat, mode, kappa, sigma, huber_delta=None, **kwargs):
    phi = w @ a_mat
    return adapt_from_state(
        u, d, phi, mu, s_mat, mode, kappa, sigma, huber_delta, **kwargs
    )

def network_msd(w, w0):
    return float(np.mean(np.sum((w - w0[:, None]) ** 2, axis=0)))

def instant_emse(u, w, w0):
    err = np.einsum("nm,mn->n", u, w - w0[:, None])
    return float(np.mean(err ** 2))

def compute_routing_rate(u, d, combination_state, kappa):
    errors = d - np.sum(u * combination_state.T, axis=1)
    return float(np.mean(np.abs(errors) > kappa))

def get_impulse_prob(config, it):
    schedule = config.get("impulse_prob_schedule")
    if schedule:
        for (start, end, p) in schedule:
            if start <= it < end:
                return p
        return schedule[-1][2]
    return config.get("impulse_prob", 0.0)


# ==========================================================
# Trial runner
# ==========================================================
def run_trial(config, rng_seed):
    rng = np.random.default_rng(rng_seed)
    n_nodes = config["n_nodes"]
    m_dim = config["m_dim"]
    adj, _ = connected_topology(n_nodes, config["radius"], rng)

    a_rel = relative_degree_weights(adj)
    c_met = metropolis_weights(adj)
    ident = np.eye(n_nodes)

    reg_covs = make_regressor_profile(n_nodes, m_dim)
    noise_std = config["noise_std"] * np.ones(n_nodes)

    if config.get("noisy_nodes"):
        noise_std[np.asarray(config["noisy_nodes"])] *= config["noisy_factor"]
    noise_var = noise_std ** 2
    a_rdv = relative_degree_variance_weights(adj, noise_var)

    w0 = rng.normal(size=m_dim)

    all_algorithms = {
        "noncoop_lms": ("atc", ident, ident, "lms"),
        "global_lms": ("atc", ident, np.ones((n_nodes, n_nodes)) / n_nodes, "lms"),
        "cta_lms_CI": ("cta", a_rel, ident, "lms"),
        "atc_lms_CI": ("atc", a_rel, ident, "lms"),
        "atc_lms_CneqI": ("atc", a_rel, c_met, "lms"),
        "atc_lms_rdv_CI": ("atc", a_rdv, ident, "lms"),
        "uniform_dmcc_atc": ("atc", a_rel, ident, "dmcc"),
        "gated_dmcc_atc": ("atc", a_rel, ident, "gated_dmcc"),
        "gated_dmcc_cta": ("cta", a_rel, ident, "gated_dmcc"),
        "gated_huber_atc": ("atc", a_rel, ident, "gated_huber"),
        "sign_error_atc": ("atc", a_rel, ident, "sign"),
        "huber_atc": ("atc", a_rel, ident, "huber"),
    }

    selected = config.get("algorithms")
    algorithms = {name: all_algorithms[name] for name in selected} if selected else all_algorithms

    if "gated_dmcc_atc" in algorithms: gate_ref_name = "gated_dmcc_atc"
    elif "gated_dmcc_cta" in algorithms: gate_ref_name = "gated_dmcc_cta"
    elif "gated_huber_atc" in algorithms: gate_ref_name = "gated_huber_atc"
    else: gate_ref_name = None

    if gate_ref_name is not None:
        gate_ref_kind, gate_ref_a_mat, _, _ = algorithms[gate_ref_name]
    else:
        gate_ref_kind, gate_ref_a_mat = None, None

    states = {name: np.zeros((m_dim, n_nodes)) for name in algorithms}
    context_states = {name: np.zeros(n_nodes) for name in algorithms}
    
    curves = {name: {"msd": [], "emse": [], "cost": [], "robust_rate": []} for name in algorithms}
    hard_rates = []
    timing = {name: 0.0 for name in algorithms}

    sigma = config.get("sigma", 1.0)
    huber_delta = config.get("huber_delta", config["kappa"])
    
    # Gate specific parameters
    gate_kwargs = {
        "gate_lambda": config.get("gate_lambda", 0.95),
        "gate_eta": config.get("gate_eta", config["kappa"]),
        "gate_context_gain": config.get("gate_context_gain", 0.5),
        "gate_threshold": config.get("gate_threshold", 1.0),
    }

    for it in range(config["iters"]):
        p_H = get_impulse_prob(config, it)
        u, d, hard_mask = sample_data(
            rng, w0, reg_covs, noise_std,
            impulse_prob=p_H,
            impulse_std=config["impulse_std"],
        )
        
        if gate_ref_name is not None:
            ref_w = states[gate_ref_name]
            gate_state = ref_w @ gate_ref_a_mat if gate_ref_kind == "cta" else ref_w
            hard_rates.append(compute_routing_rate(u, d, gate_state, config["kappa"]))
        else:
            hard_rates.append(float(hard_mask.mean()))

        for name, (kind, a_mat, s_mat, mode) in algorithms.items():
            t0 = time.perf_counter()
            if kind == "atc":
                new_w, cost, robust_rate, new_ctx = step_atc(
                    u, d, states[name], config["mu"], a_mat, s_mat, mode,
                    config["kappa"], sigma, huber_delta=huber_delta,
                    context_state=context_states[name], **gate_kwargs
                )
            else:
                new_w, cost, robust_rate, new_ctx = step_cta(
                    u, d, states[name], config["mu"], a_mat, s_mat, mode,
                    config["kappa"], sigma, huber_delta=huber_delta,
                    context_state=context_states[name], **gate_kwargs
                )
            timing[name] += time.perf_counter() - t0

            states[name] = new_w
            context_states[name] = new_ctx
            curves[name]["msd"].append(network_msd(new_w, w0))
            curves[name]["emse"].append(instant_emse(u, new_w, w0))
            curves[name]["cost"].append(float(cost))
            curves[name]["robust_rate"].append(float(robust_rate))

    return curves, float(np.mean(hard_rates)), timing

OUTLIER_TRIAL_RATIO = 50.0

def summarize(curves_list, hard_rates, timing_list, tail=100, iters=1):
    names = curves_list[0].keys()
    summary = {}
    for name in names:
        metrics = {}
        for key in ["msd", "emse", "cost", "robust_rate"]:
            vals = np.array([np.mean(c[name][key][-tail:]) for c in curves_list], dtype=float)
            metrics[f"{key}_mean"] = float(vals.mean())
            metrics[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            metrics[f"{key}_median"] = float(np.median(vals))
            med = np.median(vals)
            worst = float(vals.max())
            ratio = worst / med if med > 0 else (float("inf") if worst > 0 else 1.0)
            metrics[f"{key}_max_over_median_ratio"] = ratio
            metrics[f"{key}_has_outlier_trial"] = bool(ratio > OUTLIER_TRIAL_RATIO)
            if metrics[f"{key}_has_outlier_trial"]:
                metrics[f"{key}_outlier_trial_index"] = int(np.argmax(vals))
                metrics[f"{key}_outlier_trial_value"] = worst
                
        t_vals = np.array([t[name] for t in timing_list], dtype=float)
        metrics["wall_clock_sec_mean"] = float(t_vals.mean())
        metrics["wall_clock_sec_std"] = float(t_vals.std(ddof=1)) if len(t_vals) > 1 else 0.0
        metrics["wall_clock_sec_per_iter_mean"] = float(t_vals.mean() / max(1, iters))
        summary[name] = metrics
    summary["empirical_hard_rate"] = {
        "mean": float(np.mean(hard_rates)),
        "std": float(np.std(hard_rates, ddof=1)) if len(hard_rates) > 1 else 0.0,
    }
    return summary

def aggregate_curves(curves_list):
    names = curves_list[0].keys()
    aggregate = {}
    for name in names:
        aggregate[name] = {}
        for key in ["msd", "emse", "cost", "robust_rate"]:
            stacked = np.array([c[name][key] for c in curves_list], dtype=float)
            aggregate[name][key] = {
                "mean": stacked.mean(axis=0).tolist(),
                "std": stacked.std(axis=0, ddof=1).tolist() if stacked.shape[0] > 1 else np.zeros(stacked.shape[1]).tolist(),
            }
    return aggregate

def run_experiment(label, config):
    curves_list, hard_rates, timing_list = [], [], []
    for t in range(config["trials"]):
        curves, hard_rate, timing = run_trial(config, config["seed"] + 1009 * t)
        curves_list.append(curves)
        hard_rates.append(hard_rate)
        timing_list.append(timing)
    summary = summarize(curves_list, hard_rates, timing_list, tail=config["tail"], iters=config["iters"])
    return {
        "label": label,
        "config": {k: v for k, v in config.items() if k != "impulse_prob_schedule"},
        "summary": summary,
        "curves": aggregate_curves(curves_list)
    }


# ==========================================================
# Plotting 
# ==========================================================
def plot_experiment(exp, out_dir, schedule=None):
    label = exp["label"]
    if label == "nonstationary" and schedule is not None:
        return plot_nonstationary(exp, out_dir, schedule)

    curves = exp["curves"]
    metrics = ["emse", "msd"]
    if any(k in label for k in ("impulsive", "robust", "strong", "scalability", "nonstationary", "gated_huber")):
        metrics.append("cost")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 2.7 * len(metrics)), sharex=True, constrained_layout=True)
    if len(metrics) == 1: axes = [axes]

    for ax, metric in zip(axes, metrics):
        for name, series in curves.items():
            y = np.asarray(series[metric]["mean"], dtype=float)
            x = np.arange(1, y.size + 1)
            if metric in {"msd", "emse"}:
                ax.semilogy(x, y, linewidth=1.8, color=_get_color(name), linestyle=_get_linestyle(name),
                            marker=_get_marker(name), markevery=_markevery(y.size), markersize=5,
                            markerfacecolor="none", markeredgewidth=1.2, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
            else:
                ax.plot(x, y, linewidth=1.8, color=_get_color(name), linestyle=_get_linestyle(name),
                        marker=_get_marker(name), markevery=_markevery(y.size), markersize=5,
                        markerfacecolor="none", markeredgewidth=1.2, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.set_ylabel(metric.upper() if metric != "cost" else "Active cost\n(relative cost per neighbor eval.)")
    axes[-1].set_xlabel("Iteration")
    axes[0].legend(loc="best", fontsize=8)
    out_file = out_dir / f"{label}_curves.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return str(out_file)

def plot_nonstationary(exp, out_dir, schedule):
    curves = exp["curves"]
    selected_all = ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc", "huber_atc", "sign_error_atc", "gated_huber_atc"]
    selected_all = [n for n in selected_all if n in curves] or list(curves.keys())

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True, constrained_layout=True)

    for name in selected_all:
        y = np.asarray(curves[name]["msd"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        axes[0].semilogy(x, y, linewidth=1.8, color=_get_color(name), linestyle=_get_linestyle(name),
                         marker=_get_marker(name), markevery=_markevery(y.size), markersize=5, markerfacecolor="none",
                         markeredgewidth=1.2, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
    axes[0].set_ylabel("Network MSD"); axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].legend(loc="lower left", fontsize=8, ncol=2)

    for name in selected_all:
        y = np.asarray(curves[name]["cost"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        axes[1].plot(x, y, linewidth=1.8, color=_get_color(name), linestyle=_get_linestyle(name),
                     marker=_get_marker(name), markevery=_markevery(y.size), markersize=5, markerfacecolor="none",
                     markeredgewidth=1.2, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
    axes[1].set_ylabel("Active cost (relative cost per neighbor eval.)"); axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].legend(loc="upper right", fontsize=8, ncol=2); axes[1].set_ylim(0, 11)

    for name in ["gated_dmcc_atc", "gated_huber_atc"]:
        if name not in curves: continue
        y = np.asarray(curves[name]["robust_rate"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        axes[2].plot(x, y, linewidth=2.0, color=_get_color(name), linestyle=_get_linestyle(name),
                     marker=_get_marker(name), markevery=_markevery(y.size), markersize=6, markerfacecolor="none",
                     markeredgewidth=1.4, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))

    n_iter = len(curves[selected_all[0]]["msd"]["mean"])
    p_H_schedule = np.zeros(n_iter)
    for start, end, p in schedule: p_H_schedule[start:end] = p
    axes[2].plot(np.arange(1, n_iter + 1), p_H_schedule, "k--", linewidth=1.5, alpha=0.7, label=r"$p_H$ (impulse probability)")
    axes[2].set_ylabel(r"Gate firing rate $\bar{\Gamma}_n$"); axes[2].set_xlabel("Iteration")
    axes[2].grid(True, which="both", linestyle=":", linewidth=0.7); axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_ylim(-0.05, 1.05)

    for ax in axes:
        for (start, end, _) in schedule: ax.axvline(end, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    _annotate_regimes(axes[0], schedule, y_pos_frac=0.85)

    fig.suptitle("Non-stationary impulse probability\n" + r"(gate adapts without re-tuning $\kappa$)", fontsize=12)
    out_file = out_dir / "nonstationary_curves.png"
    fig.savefig(out_file, dpi=220, bbox_inches="tight"); plt.close(fig)
    return str(out_file)

def plot_gated_vs_standard(exp, out_dir, suffix=""):
    curves = exp["curves"]
    preferred = ["atc_lms_CI", "cta_lms_CI", "sign_error_atc", "huber_atc", "uniform_dmcc_atc", "gated_dmcc_atc", "gated_huber_atc"]
    selected = [n for n in preferred if n in curves] or list(curves.keys())

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 10.5), sharex=True, constrained_layout=True)

    for ax, metric in zip(axes[:3], ["emse", "msd", "cost"]):
        for name in selected:
            y = np.asarray(curves[name][metric]["mean"], dtype=float)
            x = np.arange(1, y.size + 1)
            if metric in {"msd", "emse"}:
                ax.semilogy(x, y, linewidth=2.0, color=_get_color(name), linestyle=_get_linestyle(name), marker=_get_marker(name),
                            markevery=_markevery(y.size), markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                            label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
            else:
                ax.plot(x, y, linewidth=2.0, color=_get_color(name), linestyle=_get_linestyle(name), marker=_get_marker(name),
                        markevery=_markevery(y.size), markersize=5, markerfacecolor="none", markeredgewidth=1.2,
                        label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.set_ylabel(metric.upper() if metric != "cost" else "Active cost\n(relative cost per neighbor eval.)")

    for name in selected:
        y = np.asarray(curves[name]["robust_rate"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        is_gated = "gated" in name
        axes[3].plot(x, y, linewidth=2.5 if is_gated else 1.2, color=_get_color(name),
                     linestyle=_get_linestyle(name) if is_gated else ":", marker=_get_marker(name),
                     markevery=_markevery(y.size), markersize=5 if is_gated else 4, markerfacecolor="none",
                     markeredgewidth=1.2, alpha=1.0 if is_gated else 0.6, label=DISPLAY_NAMES.get(name, name), zorder=_get_zorder(name))
    axes[3].set_ylabel(r"Gate / clip rate $\bar{\Gamma}$"); axes[3].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[3].set_ylim(-0.05, 1.05)

    axes[-1].set_xlabel("Iteration"); axes[0].legend(loc="best", fontsize=8); axes[3].legend(loc="best", fontsize=7, ncol=2)
    out_file = out_dir / f"agdlms_vs_standard_dlms_curves{suffix}.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_kappa_sigma_sweep(sweep_summary, out_dir):
    ratios = [row["ratio"] for row in sweep_summary]
    uniform_msd = [row["uniform_msd"] for row in sweep_summary]
    gated_msd = [row["gated_msd"] for row in sweep_summary]
    uniform_cost = [row["uniform_cost"] for row in sweep_summary]
    gated_cost = [row["gated_cost"] for row in sweep_summary]
    rel_gap = [row["relative_msd_gap"] for row in sweep_summary]
    gated_rr = [row["gated_robust_rate"] for row in sweep_summary]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True, constrained_layout=True)

    axes[0].semilogy(ratios, uniform_msd, marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].semilogy(ratios, gated_msd, marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].set_ylabel("MSD (tail mean)"); axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)

    axes[1].plot(ratios, uniform_cost, marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].plot(ratios, gated_cost, marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.0, alpha=0.6, label="ATC LMS (cost=1)")
    axes[1].set_ylabel("Active cost (per neighbor eval.)"); axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.7); axes[1].set_ylim(0, 11)

    ax3 = axes[2]
    ax3.semilogy(ratios, rel_gap, marker="^", linestyle="-", linewidth=1.8, color="#2ca02c", label="Relative MSD gap", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    ax3.set_ylabel(r"Relative MSD gap $|g-unif|/unif$"); ax3.set_xlabel(r"$\kappa / \sigma$ ratio")
    ax3.grid(True, which="both", linestyle=":", linewidth=0.7)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(ratios, gated_rr, marker="o", linestyle="--", linewidth=1.2, color="#9467bd", alpha=0.7, label=r"$\bar{\Gamma}$", markersize=6)
    ax3_twin.set_ylabel(r"Gate firing rate $\bar{\Gamma}$", color="#9467bd")
    ax3_twin.tick_params(axis='y', labelcolor="#9467bd")
    ax3_twin.set_ylim(-0.05, 1.05)

    fig.suptitle(r"Sensitivity to $\kappa / \sigma$ Ratio")
    out_file = out_dir / "kappa_sigma_sweep.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_impulse_prob_sweep(sweep_summary, out_dir):
    p_Hs = [row["p_H"] for row in sweep_summary]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.5), sharex=True, constrained_layout=True)

    alg_keys = [
        ("atc_lms_msd",     "ATC DLMS, C=I",  "#777777", "o", "-"),
        ("uniform_dmcc_msd","Uniform DMCC",   "#d62728", "s", "-"),
        ("gated_dmcc_msd",  "Gated DMCC",     "#1f77b4", "^", "--"),
        ("huber_msd",       "Huber",          "#ff7f0e", "D", "-."),
        ("sign_msd",        "Sign-Error",     "#9467bd", "v", ":"),
    ]
    for (key, lbl, color, mk, ls) in alg_keys:
        ys = [row[key] for row in sweep_summary]
        axes[0].semilogy(p_Hs, ys, marker=mk, linestyle=ls, linewidth=1.8, color=color, label=lbl, markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].set_ylabel("Tail MSD"); axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].legend(loc="best", fontsize=8)

    cost_keys = [("uniform_cost", "Uniform DMCC", "#d62728", "s", "-"), ("gated_cost", "Gated DMCC", "#1f77b4", "^", "--")]
    fixed_baselines = [("ATC LMS (cost=1)", 1.0, "#777777", ":"), ("Sign-Error (cost=2)", 2.0, "#9467bd", ":"), ("Huber (cost=4)", 4.0, "#ff7f0e", ":")]
    for (key, lbl, color, mk, ls) in cost_keys:
        if key not in sweep_summary[0]: continue
        ys = [row[key] for row in sweep_summary]
        axes[1].plot(p_Hs, ys, marker=mk, linestyle=ls, linewidth=1.8, color=color, label=lbl, markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    for (lbl, val, color, ls) in fixed_baselines:
        axes[1].axhline(val, color=color, linestyle=ls, linewidth=1.2, alpha=0.6, label=lbl)
    if "gated_cost" not in sweep_summary[0]:
        gated_cost_pred = [1 + row["gated_robust_rate"] * 9 for row in sweep_summary]
        axes[1].plot(p_Hs, gated_cost_pred, "^--", linewidth=1.5, color="#1f77b4", alpha=0.7, label="Gated DMCC (predicted)", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].set_ylabel("Tail active cost (per neighbor eval.)"); axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8); axes[1].set_ylim(0, 11)

    gated_rr = [row["gated_robust_rate"] for row in sweep_summary]
    emp_hr = [row["empirical_hard_rate"] for row in sweep_summary]
    axes[2].plot(p_Hs, gated_rr, marker="^", linestyle="--", linewidth=2.0, color="#1f77b4", label=r"Gated DMCC $\bar{\Gamma}$", markersize=8, markerfacecolor="none", markeredgewidth=1.4)
    axes[2].plot(p_Hs, emp_hr, marker="s", linestyle="-.", linewidth=1.8, color="#d62728", label="Empirical hard rate", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[2].plot(p_Hs, p_Hs, ":", color="gray", linewidth=1.0, alpha=0.7, label=r"$\bar{\Gamma}=p_H$")
    axes[2].set_xlabel(r"Impulse probability $p_H$"); axes[2].set_ylabel("Rate")
    axes[2].grid(True, which="both", linestyle=":", linewidth=0.7); axes[2].legend(loc="best", fontsize=8)
    axes[2].set_ylim(-0.05, 1.05)

    fig.suptitle(r"Sensitivity to Impulse Probability $p_H$")
    out_file = out_dir / "impulse_prob_sweep.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_network_size_sweep(sweep_summary, out_dir):
    Ns = [row["N"] for row in sweep_summary]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.5), sharex=True, constrained_layout=True)

    alg_keys = [("atc_lms_msd", "ATC DLMS, C=I", "#777777", "o", "-"), ("uniform_dmcc_msd", "Uniform DMCC", "#d62728", "s", "-"), ("gated_dmcc_msd", "Gated DMCC", "#1f77b4", "^", "--")]
    for (key, lbl, color, mk, ls) in alg_keys:
        ys = [row[key] for row in sweep_summary]
        axes[0].semilogy(Ns, ys, marker=mk, linestyle=ls, linewidth=1.8, color=color, label=lbl, markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].set_ylabel("Tail MSD"); axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].plot(Ns, [row["uniform_cost"] for row in sweep_summary], marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC cost", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].plot(Ns, [row["gated_cost"] for row in sweep_summary], marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC cost", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.0, alpha=0.6, label="ATC LMS (cost=1)")
    axes[1].set_ylabel("Tail active cost"); axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].legend(loc="best", fontsize=8); axes[1].set_ylim(0, 11)

    axes[2].plot(Ns, [row["uniform_wall_clock_sec"] for row in sweep_summary], marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC time", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[2].plot(Ns, [row["gated_wall_clock_sec"] for row in sweep_summary], marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC time", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[2].set_xlabel(r"Network size $N$ (with $M=2$ fixed)"); axes[2].set_ylabel("Wall-clock time per trial (s)")
    axes[2].grid(True, which="both", linestyle=":", linewidth=0.7); axes[2].legend(loc="best", fontsize=8)

    fig.suptitle(r"Scalability with Network Size $N$ ($M=2$ fixed)")
    out_file = out_dir / "network_size_sweep.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_wall_clock(exp, out_dir, suffix=""):
    summary = exp["summary"]
    names = [n for n in summary.keys() if n != "empirical_hard_rate"]
    names.sort(key=lambda n: summary[n].get("cost_mean", 0))
    times = [summary[n]["wall_clock_sec_mean"] for n in names]
    errs = [summary[n]["wall_clock_sec_std"] for n in names]
    labels = [DISPLAY_NAMES.get(n, n) for n in names]
    colors = [_get_color(n) for n in names]
    hatch_cycle = ["", "//", "xx", "\\\\", "..", "++", "oo", "**", "--", "||"]
    hatches = [hatch_cycle[i % len(hatch_cycle)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    x = np.arange(len(names))
    for xi, t, e, c, h in zip(x, times, errs, colors, hatches):
        ax.bar(xi, t, yerr=e, capsize=3, color=c, alpha=0.85, edgecolor="black", linewidth=0.7, hatch=h)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time per trial (s)"); ax.grid(True, axis="y", linestyle=":", linewidth=0.7)
    title = "Wall-clock Time per Algorithm"
    if suffix: title += f" — {suffix}"
    ax.set_title(title)
    out_file = out_dir / f"wall_clock{suffix}.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_impulse_rate_comparison(exps, out_dir):
    selected = ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc", "huber_atc", "sign_error_atc"]
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 2.6 * n), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    n_pH = max(len(exps), 1)
    colors = [cmap(i / max(1, n_pH - 1)) for i in range(n_pH)]
    linestyle_cycle = ["-", "--", "-.", ":"]
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]

    for ax, name in zip(axes, selected):
        for i, exp in enumerate(exps):
            if name not in exp["curves"]: continue
            y = np.asarray(exp["curves"][name]["msd"]["mean"], dtype=float)
            x = np.arange(1, y.size + 1)
            p_H = exp["config"]["impulse_prob"]
            ax.semilogy(x, y, linewidth=1.8, color=colors[i % len(colors)],
                        linestyle=linestyle_cycle[i % len(linestyle_cycle)],
                        marker=marker_cycle[i % len(marker_cycle)],
                        markevery=_markevery(y.size, target_markers=8), markersize=5, markerfacecolor="none",
                        markeredgewidth=1.1, label=fr"$p_H={p_H:g}$")
        ax.grid(True, which="both", linestyle=":", linewidth=0.7)
        ax.set_ylabel("MSD"); ax.set_title(DISPLAY_NAMES.get(name, name), fontsize=10)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle("Effect of Impulse Rate $p_H$ on Convergence")
    out_file = out_dir / "impulse_rate_comparison.png"
    fig.savefig(out_file, dpi=220); plt.close(fig)
    return str(out_file)

def plot_decision_boundary(out_dir, kappa=1.0, eta=1.0, gate_context_gain=0.5, gate_threshold=1.0):
    """Plots the Multiplicative Context-Aware Gate decision boundary."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

    e_range = np.linspace(0, 3 * kappa, 300)
    sigma_e_range = np.linspace(0, 5 * eta, 300)
    E_grid, S_grid = np.meshgrid(e_range, sigma_e_range)

    # Context in [0, 1]
    Context_grid = 1.0 - np.exp(-S_grid / (eta + 1e-12))
    
    # Multiplicative hardness score
    H_grid = (E_grid / kappa) * (1.0 + gate_context_gain * Context_grid)

    contour = ax.contourf(E_grid / kappa, S_grid / eta, H_grid,
                          levels=[0, 1, 2, 3, 4, 5], cmap='RdYlGn_r', alpha=0.7)
    ax.contour(E_grid / kappa, S_grid / eta, H_grid,
               levels=[gate_threshold], colors='black', linewidths=2, linestyles='--')
    
    # Naive gate boundary is just |e|/kappa = threshold
    ax.axvline(x=gate_threshold, color='blue', linewidth=2, linestyle='-',
               label=f'Naive gate threshold (|e|/κ = {gate_threshold})')

    ax.set_xlabel('|e| / κ')
    ax.set_ylabel('σ_e / η  (Context: Error History)')
    ax.set_title('Decision Boundary: Multiplicative Context-Aware Gate vs Naive Gate\n'
                 '(Black dashed = Context-aware boundary, Blue line = Naive boundary)')
    ax.legend(fontsize=9)
    plt.colorbar(contour, ax=ax, label='Hardness score h')
    
    out_file = out_dir / "decision_boundary.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return str(out_file)


# ==========================================================
# Sweep helpers
# ==========================================================
def run_kappa_sigma_sweep(base_config, ratios, sigma=1.0):
    sweep_summary = []
    sweep_curves = {}
    for ratio in ratios:
        cfg = dict(base_config)
        cfg["sigma"] = sigma
        cfg["kappa"] = ratio * sigma
        cfg["algorithms"] = ["uniform_dmcc_atc", "gated_dmcc_atc"]
        exp = run_experiment(f"kappa_sigma_ratio_{ratio:g}", cfg)
        summary = exp["summary"]
        uniform_msd = summary["uniform_dmcc_atc"]["msd_mean"]
        gated_msd = summary["gated_dmcc_atc"]["msd_mean"]
        sweep_summary.append({
            "ratio": ratio, "kappa": cfg["kappa"], "sigma": sigma,
            "uniform_msd": uniform_msd, "gated_msd": gated_msd,
            "relative_msd_gap": abs(gated_msd - uniform_msd) / max(uniform_msd, 1e-12),
            "uniform_cost": summary["uniform_dmcc_atc"]["cost_mean"],
            "gated_cost": summary["gated_dmcc_atc"]["cost_mean"],
            "gated_robust_rate": summary["gated_dmcc_atc"]["robust_rate_mean"],
        })
        sweep_curves[ratio] = exp["curves"]
    return sweep_summary, sweep_curves

def run_impulse_prob_sweep(base_config, p_H_values):
    sweep_summary = []
    sweep_curves = {}
    for p in p_H_values:
        cfg = dict(base_config)
        cfg["impulse_prob"] = p
        cfg["impulse_prob_schedule"] = None
        cfg["algorithms"] = ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc", "huber_atc", "sign_error_atc"]
        exp = run_experiment(f"impulse_prob_{p:g}", cfg)
        summary = exp["summary"]
        sweep_summary.append({
            "p_H": p,
            "atc_lms_msd": summary["atc_lms_CI"]["msd_mean"],
            "uniform_dmcc_msd": summary["uniform_dmcc_atc"]["msd_mean"],
            "gated_dmcc_msd": summary["gated_dmcc_atc"]["msd_mean"],
            "huber_msd": summary["huber_atc"]["msd_mean"],
            "sign_msd": summary["sign_error_atc"]["msd_mean"],
            "uniform_cost": summary["uniform_dmcc_atc"]["cost_mean"],
            "gated_cost":   summary["gated_dmcc_atc"]["cost_mean"],
            "gated_robust_rate": summary["gated_dmcc_atc"]["robust_rate_mean"],
            "empirical_hard_rate": summary["empirical_hard_rate"]["mean"],
        })
        sweep_curves[p] = exp["curves"]
    return sweep_summary, sweep_curves

def run_network_size_sweep(base_config, n_values, m_dim=2):
    sweep_summary = []
    sweep_curves = {}
    for n in n_values:
        cfg = dict(base_config)
        cfg["n_nodes"] = n
        cfg["m_dim"] = m_dim
        cfg["radius"] = max(0.20, 0.34 * (30.0 / max(1, n)) ** 0.5)
        orig_noisy = cfg.get("noisy_nodes") or []
        if orig_noisy:
            scaled = sorted({min(n - 1, max(0, int(idx))) for idx in orig_noisy})
            cfg["noisy_nodes"] = scaled
        cfg["algorithms"] = ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc"]
        exp = run_experiment(f"network_size_N{n}", cfg)
        summary = exp["summary"]
        sweep_summary.append({
            "N": n, "M": m_dim,
            "atc_lms_msd": summary["atc_lms_CI"]["msd_mean"],
            "uniform_dmcc_msd": summary["uniform_dmcc_atc"]["msd_mean"],
            "gated_dmcc_msd": summary["gated_dmcc_atc"]["msd_mean"],
            "uniform_cost": summary["uniform_dmcc_atc"]["cost_mean"],
            "gated_cost": summary["gated_dmcc_atc"]["cost_mean"],
            "gated_robust_rate": summary["gated_dmcc_atc"]["robust_rate_mean"],
            "gated_wall_clock_sec": summary["gated_dmcc_atc"]["wall_clock_sec_mean"],
            "uniform_wall_clock_sec": summary["uniform_dmcc_atc"]["wall_clock_sec_mean"],
        })
        sweep_curves[n] = exp["curves"]
    return sweep_summary, sweep_curves

# Lambda_c (Context Gain) Sweep
def run_lambda_c_sweep(base_config, lambda_c_values):
    sweep_summary = []
    sweep_curves = {}
    for lam_c in lambda_c_values:
        cfg = dict(base_config)
        cfg["gate_context_gain"] = lam_c
        cfg["algorithms"] = ["uniform_dmcc_atc", "gated_dmcc_atc"]
        exp = run_experiment(f"lambda_c_{lam_c:g}", cfg)
        summary = exp["summary"]
        uniform_msd = summary["uniform_dmcc_atc"]["msd_mean"]
        gated_msd = summary["gated_dmcc_atc"]["msd_mean"]
        sweep_summary.append({
            "lambda_c": lam_c,
            "uniform_msd": uniform_msd,
            "gated_msd": gated_msd,
            "relative_msd_gap": abs(gated_msd - uniform_msd) / max(uniform_msd, 1e-12),
            "uniform_cost": summary["uniform_dmcc_atc"]["cost_mean"],
            "gated_cost": summary["gated_dmcc_atc"]["cost_mean"],
            "gated_robust_rate": summary["gated_dmcc_atc"]["robust_rate_mean"],
        })
        sweep_curves[lam_c] = exp["curves"]
    return sweep_summary, sweep_curves

# Lambda_c (Context Gain) Sweep Plot
def plot_lambda_c_sweep(sweep_summary, out_dir):
    lam_cs = [row["lambda_c"] for row in sweep_summary]
    uniform_msd = [row["uniform_msd"] for row in sweep_summary]
    gated_msd = [row["gated_msd"] for row in sweep_summary]
    uniform_cost = [row["uniform_cost"] for row in sweep_summary]
    gated_cost = [row["gated_cost"] for row in sweep_summary]
    gated_rr = [row["gated_robust_rate"] for row in sweep_summary]
    rel_gap = [row["relative_msd_gap"] for row in sweep_summary]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True, constrained_layout=True)

    # Plot 1: MSD
    axes[0].semilogy(lam_cs, uniform_msd, marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].semilogy(lam_cs, gated_msd, marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[0].set_ylabel("Steady-state MSD")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].set_title(r"Sensitivity to Context Gain $\lambda_c$ ($\lambda_c=0$ is static threshold)")

    # Plot 2: Cost
    axes[1].plot(lam_cs, uniform_cost, marker="s", linestyle="-", linewidth=1.8, color="#d62728", label="Uniform DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].plot(lam_cs, gated_cost, marker="^", linestyle="--", linewidth=1.8, color="#1f77b4", label="Gated DMCC", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    axes[1].axhline(1.0, color="#777777", linestyle=":", linewidth=1.0, alpha=0.6, label="ATC LMS (cost=1)")
    axes[1].set_ylabel("Active cost")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].set_ylim(0, 11)

    # Plot 3: Gate Firing Rate & Relative MSD Gap
    ax3 = axes[2]
    ax3.plot(lam_cs, gated_rr, marker="o", linestyle="--", linewidth=1.8, color="#9467bd", label=r"Gate firing rate $\bar{\Gamma}$", markersize=7, markerfacecolor="none", markeredgewidth=1.3)
    ax3.set_xlabel(r"Context-aware gain $\lambda_c$")
    ax3.set_ylabel(r"Gate firing rate $\bar{\Gamma}$", color="#9467bd")
    ax3.tick_params(axis='y', labelcolor="#9467bd")
    ax3.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax3.set_ylim(-0.05, 1.05)
    
    ax3_twin = ax3.twinx()
    ax3_twin.semilogy(lam_cs, rel_gap, marker="^", linestyle="-", linewidth=1.2, color="#2ca02c", alpha=0.7, label="Rel. MSD gap", markersize=6)
    ax3_twin.set_ylabel("Relative MSD gap", color="#2ca02c")
    ax3_twin.tick_params(axis='y', labelcolor="#2ca02c")

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)

    out_file = out_dir / "lambda_c_sweep.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return str(out_file)

# Lambda_c impact on Non-Stationary Environment
def plot_lambda_c_nonstationary_curves(sweep_curves, lambda_c_values, out_dir, schedule):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(lambda_c_values)-1)) for i in range(len(lambda_c_values))]

    # Plot 1: MSD Trajectories
    for i, lam_c in enumerate(lambda_c_values):
        y = np.asarray(sweep_curves[lam_c]["gated_dmcc_atc"]["msd"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        axes[0].semilogy(x, y, linewidth=2.0, color=colors[i], linestyle="--", 
                         label=fr"$\lambda_c = {lam_c:g}$")
    
    for (start, end, p) in schedule:
        axes[0].axvline(end, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    _annotate_regimes(axes[0], schedule, y_pos_frac=0.85)
    axes[0].set_ylabel("Network MSD")
    axes[0].set_title(r"Impact of Context Gain ($\lambda_c$) on Non-Stationary Impulses")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[0].legend(loc="lower left", fontsize=8)

    # Plot 2: Gate Firing Rate Trajectories
    for i, lam_c in enumerate(lambda_c_values):
        y = np.asarray(sweep_curves[lam_c]["gated_dmcc_atc"]["robust_rate"]["mean"], dtype=float)
        x = np.arange(1, y.size + 1)
        axes[1].plot(x, y, linewidth=2.0, color=colors[i], linestyle="--", 
                     label=fr"$\lambda_c = {lam_c:g}$")
        
    n_iter = len(y)
    p_H_schedule = np.zeros(n_iter)
    for start, end, p in schedule: p_H_schedule[start:end] = p
    axes[1].plot(np.arange(1, n_iter + 1), p_H_schedule, "k-", linewidth=2.5, alpha=0.5, label=r"$p_H$ (Target Impulse Prob.)")
    
    for (start, end, p) in schedule:
        axes[1].axvline(end, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    axes[1].set_ylabel(r"Gate firing rate $\bar{\Gamma}_n$")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.7)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylim(-0.05, 0.65)

    out_file = out_dir / "lambda_c_nonstationary_curves.png"
    fig.savefig(out_file, dpi=220)
    plt.close(fig)
    return str(out_file)

# ==========================================================
# Main
# ==========================================================
def main():
    base = {
        "n_nodes": 30, "m_dim": 2, "radius": 0.34, "iters": 2000, "tail": 500,
        "trials": 20, "mu": 0.05, "noise_std": 0.05, "sigma": 1.0,
        "impulse_prob": 0.0, "impulse_std": 0.0, "kappa": 0.20, "seed": 4317,
        "noisy_nodes": [3, 14, 25], "noisy_factor": 1.0,
        # Context Gate Parameters
        "gate_lambda": 0.95,
        "gate_eta": 1.0,
        "gate_context_gain": 0.5,
        "gate_threshold": 1.0,
        "algorithms": ["noncoop_lms", "global_lms", "cta_lms_CI", "atc_lms_CI", "atc_lms_CneqI"],
    }

    impulsive = dict(base)
    impulsive.update({
        "iters": 2000, "tail": 500, "trials": 20,
        "impulse_prob": 0.05, "impulse_std": 5.0, "seed": 9013,
        "algorithms": ["atc_lms_CI", "cta_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc", "gated_dmcc_cta"],
    })

    impulsive_p10 = dict(impulsive)
    impulsive_p10.update({"impulse_prob": 0.10, "seed": 99821})

    noisy = dict(base)
    noisy.update({
        "iters": 350, "tail": 80, "trials": 20, "seed": 12011,
        "noisy_factor": 50.0,
        "algorithms": ["noncoop_lms", "atc_lms_CI", "atc_lms_rdv_CI", "global_lms"],
    })

    robust_baselines = dict(impulsive)
    robust_baselines.update({
        "seed": 11027,
        "algorithms": ["atc_lms_CI", "sign_error_atc", "huber_atc", "uniform_dmcc_atc", "gated_dmcc_atc", "gated_huber_atc"],
        "huber_delta": 0.5,
    })

    scalability = dict(impulsive)
    scalability.update({
        "n_nodes": 50, "m_dim": 2, "iters": 400, "tail": 80, "trials": 20,
        "seed": 33441, "radius": 0.28,
        "algorithms": ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc"],
    })

    strong_impulse = dict(impulsive)
    strong_impulse.update({
        "impulse_prob": 0.08, "impulse_std": 20.0, "seed": 55661,
        "algorithms": ["atc_lms_CI", "sign_error_atc", "huber_atc", "uniform_dmcc_atc", "gated_dmcc_atc", "gated_huber_atc"],
    })

    nonstationary = dict(impulsive)
    nonstationary.update({
        "seed": 77001, "iters": 2000, "tail": 200,
        "impulse_prob": 0.0,
        "impulse_prob_schedule": [(0, 600, 0.05), (600, 1400, 0.30), (1400, 2000, 0.05)],
        "algorithms": ["atc_lms_CI", "uniform_dmcc_atc", "gated_dmcc_atc", "huber_atc", "sign_error_atc", "gated_huber_atc"],
    })

    results = [
        run_experiment("base_gaussian_dlms", base),
        run_experiment("impulsive_asymmetric_dlms", impulsive),
        run_experiment("impulsive_p10_dlms", impulsive_p10),
        run_experiment("noisy_node_weighting", noisy),
        run_experiment("robust_baselines", robust_baselines),
        run_experiment("scalability_N50_M2", scalability),
        run_experiment("strong_impulse", strong_impulse),
        run_experiment("nonstationary", nonstationary),
    ]

    out_dir = Path("code_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "routeDMCC_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    figure_files = []
    for exp in results:
        if exp["label"] == "nonstationary":
            figure_files.append(plot_experiment(exp, out_dir, schedule=nonstationary["impulse_prob_schedule"]))
        else:
            figure_files.append(plot_experiment(exp, out_dir))
            
    figure_files.append(plot_gated_vs_standard(results[1], out_dir, suffix="_p05"))
    figure_files.append(plot_gated_vs_standard(results[2], out_dir, suffix="_p10"))
    figure_files.append(plot_gated_vs_standard(results[4], out_dir, suffix="_baselines"))
    figure_files.append(plot_wall_clock(results[4], out_dir, suffix="_baselines"))
    figure_files.append(plot_nonstationary(results[7], out_dir, nonstationary["impulse_prob_schedule"]))

    # Generate the Multiplicative Decision Boundary Plot
    figure_files.append(plot_decision_boundary(
        out_dir, 
        kappa=base["kappa"], 
        eta=base["gate_eta"], 
        gate_context_gain=base["gate_context_gain"], 
        gate_threshold=base["gate_threshold"]
    ))

    # kappa/sigma sweep
    sweep_base = dict(impulsive)
    sweep_base.update({"trials": 20, "seed": 20231})
    sweep_ratios = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    sweep_summary, _ = run_kappa_sigma_sweep(sweep_base, sweep_ratios, sigma=1.0)
    sweep_file = out_dir / "kappa_sigma_sweep_summary.json"
    sweep_file.write_text(json.dumps(sweep_summary, indent=2), encoding="utf-8")
    figure_files.append(plot_kappa_sigma_sweep(sweep_summary, out_dir))

    # Real impulse-probability sweep
    p_H_sweep_base = dict(impulsive)
    p_H_sweep_base.update({"trials": 20, "seed": 40111})
    p_H_values = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    p_H_summary, p_H_curves = run_impulse_prob_sweep(p_H_sweep_base, p_H_values)
    p_H_file = out_dir / "impulse_prob_sweep_summary.json"
    p_H_file.write_text(json.dumps(p_H_summary, indent=2), encoding="utf-8")
    figure_files.append(plot_impulse_prob_sweep(p_H_summary, out_dir))
    
    sweep_exps = [{"label": f"p_H={p:g}", "config": {"impulse_prob": p}, "curves": p_H_curves[p]} for p in p_H_values]
    figure_files.append(plot_impulse_rate_comparison(sweep_exps, out_dir))

    # Network-size sweep (M fixed = 2)
    N_sweep_base = dict(impulsive)
    N_sweep_base.update({"trials": 20, "seed": 51277})
    N_values = [10, 20, 30, 60, 80, 100]
    N_summary, _ = run_network_size_sweep(N_sweep_base, N_values, m_dim=2)
    N_file = out_dir / "network_size_sweep_summary.json"
    N_file.write_text(json.dumps(N_summary, indent=2), encoding="utf-8")
    figure_files.append(plot_network_size_sweep(N_summary, out_dir))

    # NEW: Context gain (lambda_c) sweep
    lambda_c_sweep_base = dict(impulsive)
    lambda_c_sweep_base.update({"trials": 20, "seed": 61277})
    lambda_c_values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    lambda_c_summary, _ = run_lambda_c_sweep(lambda_c_sweep_base, lambda_c_values)
    lambda_c_file = out_dir / "lambda_c_sweep_summary.json"
    lambda_c_file.write_text(json.dumps(lambda_c_summary, indent=2), encoding="utf-8")
    figure_files.append(plot_lambda_c_sweep(lambda_c_summary, out_dir))

    # Context gain (lambda_c) sweep IN NON-STATIONARY ENVIRONMENT
    ns_lambda_c_base = dict(nonstationary)
    ns_lambda_c_base.update({"trials": 10, "seed": 61277, "tail": 200}) # کم کردن trials برای سرعت بیشتر در غیرایستا
    ns_lambda_c_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    ns_lambda_c_summary, ns_lambda_c_curves = run_lambda_c_sweep(ns_lambda_c_base, ns_lambda_c_values)
    
    # plot_lambda_c_nonstationary_curves
    figure_files.append(plot_lambda_c_nonstationary_curves(
        ns_lambda_c_curves, ns_lambda_c_values, out_dir, nonstationary["impulse_prob_schedule"]
    ))

    print("\n" + "=" * 72)
    print("Experiments included:")
    print("  - base_gaussian_dlms")
    print("  - impulsive_asymmetric_dlms (p_H=0.05)")
    print("  - impulsive_p10_dlms (p_H=0.10)")
    print("  - noisy_node_weighting")
    print("  - robust_baselines (Sign-Error + Huber + Uniform/Gated DMCC + Gated Huber)")
    print("  - scalability_N50_M2  (M held at 2)")
    print("  - strong_impulse (p_H=0.08, sigma_imp=20)")
    print("  - nonstationary (p_H schedule 0.05 -> 0.30 -> 0.05)")
    print("  - kappa/sigma sensitivity sweep")
    print("  - impulse-probability sweep  (p_H in {0.01..0.50})")
    print("  - network-size sweep  (N in {10,20,30,60,80,100}, M=2)")
    print("  - Multiplicative Context-Aware Decision Boundary Plot")
    print("  - lambda_c (context gain) sweep (lambda_c in {0.0..2.0})")
    print("=" * 72 + "\n")

    for exp in results:
        print(f"\n== {exp['label']} ==")
        print("empirical_hard_rate:", exp["summary"]["empirical_hard_rate"])
        for name, metrics in exp["summary"].items():
            if name == "empirical_hard_rate": continue
            print(
                f"{name:22s} MSD={metrics['msd_mean']:.3e} "
                f"± {metrics['msd_std']:.2e}   "
                f"EMSE={metrics['emse_mean']:.3e} "
                f"± {metrics['emse_std']:.2e}   "
                f"Cost={metrics['cost_mean']:.3f} "
                f"± {metrics['cost_std']:.2f}   "
                f"Robust={metrics['robust_rate_mean']:.3f} "
                f"± {metrics['robust_rate_std']:.3f}   "
                f"Time={metrics['wall_clock_sec_mean']:.3f}s"
            )
            if metrics.get("msd_has_outlier_trial"):
                print(
                    f"    WARNING: {name} in '{exp['label']}' has an outlier "
                    f"trial (index {metrics['msd_outlier_trial_index']}) whose "
                    f"tail MSD ({metrics['msd_outlier_trial_value']:.3e}) is "
                    f"{metrics['msd_max_over_median_ratio']:.1f}x the median "
                    f"({metrics['msd_median']:.3e}). This usually means that "
                    f"trial's true weight w0 had an unusually large norm and "
                    f"the MCC-type kernel (sigma) stalled its convergence "
                    f"before 'tail' was reached -- the mean above is skewed "
                    f"by this single trial; msd_median is the robust figure."
                )

    print("\n" + "=" * 72)
    print("== kappa/sigma sensitivity sweep ==")
    print(f"{'ratio':>8s} {'kappa':>8s} {'unif_MSD':>12s} {'gated_MSD':>12s} "
          f"{'rel_gap':>10s} {'unif_cost':>10s} {'gated_cost':>11s} {'gated_rr':>9s}")
    for row in sweep_summary:
        print(
            f"{row['ratio']:8.3g} {row['kappa']:8.3g} "
            f"{row['uniform_msd']:12.6g} {row['gated_msd']:12.6g} "
            f"{row['relative_msd_gap']:10.4g} {row['uniform_cost']:10.3f} "
            f"{row['gated_cost']:11.3f} {row['gated_robust_rate']:9.3f}"
        )

    print("\n== impulse-probability sweep ==")
    print(f"{'p_H':>6s} {'ATC_LMS':>12s} {'unif_DMCC':>12s} {'gated_DMCC':>12s} "
          f"{'huber':>12s} {'sign':>12s} {'unif_cost':>10s} {'gated_cost':>11s} "
          f"{'gated_rr':>9s} {'emp_hr':>9s}")
    for row in p_H_summary:
        print(
            f"{row['p_H']:6.3g} {row['atc_lms_msd']:12.4g} "
            f"{row['uniform_dmcc_msd']:12.4g} {row['gated_dmcc_msd']:12.4g} "
            f"{row['huber_msd']:12.4g} {row['sign_msd']:12.4g} "
            f"{row['uniform_cost']:10.3f} {row['gated_cost']:11.3f} "
            f"{row['gated_robust_rate']:9.3f} {row['empirical_hard_rate']:9.3f}"
        )

    print("\n== network-size sweep (M=2 fixed) ==")
    print(f"{'N':>5s} {'M':>3s} {'ATC_LMS':>12s} {'unif_DMCC':>12s} "
          f"{'gated_DMCC':>12s} {'unif_cost':>10s} {'gated_cost':>11s} "
          f"{'gated_rr':>9s} {'unif_t(s)':>10s} {'gated_t(s)':>11s}")
    for row in N_summary:
        print(
            f"{row['N']:5d} {row['M']:3d} {row['atc_lms_msd']:12.4g} "
            f"{row['uniform_dmcc_msd']:12.4g} {row['gated_dmcc_msd']:12.4g} "
            f"{row['uniform_cost']:10.3f} {row['gated_cost']:11.3f} "
            f"{row['gated_robust_rate']:9.3f} "
            f"{row['uniform_wall_clock_sec']:10.3f} "
            f"{row['gated_wall_clock_sec']:11.3f}"
        )

    print("\n== lambda_c (context gain) sweep ==")
    print(f"{'lam_c':>6s} {'unif_MSD':>12s} {'gated_MSD':>12s} "
          f"{'rel_gap':>10s} {'unif_cost':>10s} {'gated_cost':>11s} {'gated_rr':>9s}")
    for row in lambda_c_summary:
        print(
            f"{row['lambda_c']:6.3g} {row['uniform_msd']:12.4g} "
            f"{row['gated_msd']:12.4g} {row['relative_msd_gap']:10.4g} "
            f"{row['uniform_cost']:10.3f} {row['gated_cost']:11.3f} "
            f"{row['gated_robust_rate']:9.3f}"
        )

    print("\nFigures saved to:")
    for figure_file in figure_files:
        print(f" - {figure_file}")


if __name__ == "__main__":
    main()