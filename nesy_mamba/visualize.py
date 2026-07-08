"""
Visualization Utilities for NeSy Mamba.

Includes:
  - Training curve plots (loss, accuracy, fidelity)
  - Slot activation heatmaps
  - Ablation comparison charts
  - Explanation trace visualization

Requires matplotlib: pip install matplotlib
"""

import os
import json
import torch
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


# ── Training Curves ─────────────────────────────────────────────────

def plot_training_curves(
    history: dict,
    title: str = "NeSy Mamba Training",
    save_path: str | None = None,
):
    """
    Plot training and validation curves.

    Args:
        history: dict with "train" and "val" keys, each a list of
                 dicts with "loss", "accuracy", "logic_fidelity", etc.
        title: plot title
        save_path: if provided, save figure to this path
    """
    _check_mpl()

    train_h = history["train"]
    val_h = history["val"]
    epochs = range(1, len(train_h) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [t["loss"] for t in train_h], "b-", label="Train")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, [t["accuracy"] for t in train_h], "b-", label="Train")
    ax.plot(epochs, [v["accuracy"] for v in val_h], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Logic Fidelity
    ax = axes[1, 0]
    ax.plot(epochs, [t["logic_fidelity"] for t in train_h], "b-", label="Train")
    ax.plot(epochs, [v["logic_fidelity"] for v in val_h], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Logic Fidelity")
    ax.set_title("Logic Fidelity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Slot Fidelity
    ax = axes[1, 1]
    ax.plot(epochs, [t.get("slot_fidelity", 0) for t in train_h], "b-", label="Train")
    ax.plot(epochs, [v.get("slot_fidelity", 0) for v in val_h], "r--", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Slot Fidelity")
    ax.set_title("Slot Fidelity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Training curves saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ── Slot Activation Heatmap ─────────────────────────────────────────

def plot_slot_heatmap(
    slot_history: torch.Tensor,
    rule_names: list[str] | None = None,
    title: str = "Slot Activations Over Time",
    save_path: str | None = None,
    example_idx: int = 0,
):
    """
    Plot slot activation trajectory for a single example.

    Args:
        slot_history: (B, L, K) tensor of slot values over time
        rule_names: list of K rule names
        title: plot title
        save_path: save path (optional)
        example_idx: which example in the batch to plot
    """
    _check_mpl()

    data = slot_history[example_idx].detach().cpu().numpy()  # (L, K)
    L, K = data.shape

    if rule_names is None:
        rule_names = [f"Slot {i}" for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(8, L * 0.3), max(4, K * 0.5)))
    im = ax.imshow(data.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Slot / Rule")
    ax.set_yticks(range(K))
    ax.set_yticklabels(rule_names, fontsize=8)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Slot heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ── Ablation Comparison Chart ───────────────────────────────────────

def plot_ablation_comparison(
    results: list[dict] | str,
    save_path: str | None = None,
):
    """
    Plot bar chart comparing ablation variants.

    Args:
        results: list of dicts with "variant" and "final_val" keys,
                 OR path to ablation_results.json
        save_path: save path (optional)
    """
    _check_mpl()

    if isinstance(results, str):
        with open(results) as f:
            results = json.load(f)

    variants = [r["variant"] for r in results]
    metrics = {
        "Accuracy": [r["final_val"]["accuracy"] for r in results],
        "Logic Fidelity": [r["final_val"]["logic_fidelity"] for r in results],
        "Slot Fidelity": [r["final_val"].get("slot_fidelity", 0) for r in results],
    }

    x = np.arange(len(variants))
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax.bar(x + i * width, values, width, label=metric_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Variant")
    ax.set_ylabel("Score")
    ax.set_title("NeSy Mamba — Ablation Study", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Ablation chart saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ── Explanation Trace ───────────────────────────────────────────────

def plot_explanation_trace(
    explanation: dict,
    title: str = "Self-Explanation Trace",
    save_path: str | None = None,
):
    """
    Visualize a model's self-explanation as a timeline.

    Args:
        explanation: dict from model.explain() with "answer", "confidence", "trace"
        save_path: save path (optional)
    """
    _check_mpl()

    trace = explanation.get("trace", [])
    if not trace:
        print("  No trace to visualize (no slots fired).")
        return

    steps = [t["step"] for t in trace]
    values = [t["value"] for t in trace]
    names = [t["rule"] for t in trace]

    fig, ax = plt.subplots(figsize=(max(6, len(trace) * 1.5), 4))

    colors = plt.cm.Set2(np.linspace(0, 1, len(trace)))
    bars = ax.barh(range(len(trace)), values, color=colors)

    ax.set_yticks(range(len(trace)))
    ax.set_yticklabels([f"Step {s}: {n}" for s, n in zip(steps, names)], fontsize=9)
    ax.set_xlabel("Activation Value")
    ax.set_xlim(0, 1.1)
    ax.set_title(
        f"{title}\nAnswer: {explanation['answer']} "
        f"(confidence={explanation['confidence']})",
        fontsize=11,
    )

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Trace plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ── Batch Visualizations ───────────────────────────────────────────

def generate_all_plots(
    history: dict,
    model=None,
    val_loader=None,
    rule_names: list[str] | None = None,
    output_dir: str = "figures",
    variant: str = "full",
):
    """Generate all standard plots and save to output_dir."""
    _check_mpl()
    os.makedirs(output_dir, exist_ok=True)

    # Training curves
    plot_training_curves(
        history,
        title=f"NeSy Mamba — {variant.upper()}",
        save_path=os.path.join(output_dir, f"training_{variant}.png"),
    )

    # Slot heatmap (if model provided)
    if model is not None and val_loader is not None and model.slot_gate is not None:
        sample_batch = next(iter(val_loader))
        device = next(model.parameters()).device
        sample_ids = sample_batch["input_ids"][:1].to(device)

        with torch.no_grad():
            model.eval()
            result = model(sample_ids)

        if result["slot_history"] is not None:
            plot_slot_heatmap(
                result["slot_history"],
                rule_names=rule_names,
                title=f"Slot Activations — {variant.upper()}",
                save_path=os.path.join(output_dir, f"slots_{variant}.png"),
            )

        # Explanation trace
        explanations = model.explain(sample_ids, rule_names)
        if explanations and explanations[0]["trace"]:
            plot_explanation_trace(
                explanations[0],
                title=f"Explanation — {variant.upper()}",
                save_path=os.path.join(output_dir, f"trace_{variant}.png"),
            )

    print(f"  All plots saved to {output_dir}/")


# ── Firing Order vs Proof Depth Analysis ────────────────────────────

def compute_firing_depth_correlation(
    firing_orders: torch.Tensor,  # (N, K) — firing timesteps (-1 = unfired)
    proof_depths: torch.Tensor,   # (N,)   — proof depth per example
    threshold: int = 0,           # ignore slots that never fired
) -> dict:
    """
    Compute correlation between mean slot firing time and proof depth.

    For each example, the mean firing timestep across fired slots is
    computed. Then we measure Spearman rank correlation between this
    mean firing time and the proof depth.

    Returns:
        dict with keys:
            - "spearman_r": Spearman rank correlation coefficient
            - "spearman_p": p-value
            - "depth_to_mean_firing": dict mapping depth → mean firing time
            - "n_examples": number of valid examples used
    """
    N = firing_orders.shape[0]
    fo_np = firing_orders.cpu().numpy()
    pd_np = proof_depths.cpu().numpy()

    mean_firings = []
    valid_depths = []

    for i in range(N):
        fired_mask = fo_np[i] > threshold
        if fired_mask.any():
            mean_firings.append(float(fo_np[i][fired_mask].mean()))
            valid_depths.append(int(pd_np[i]))

    if len(mean_firings) < 3:
        return {
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "depth_to_mean_firing": {},
            "n_examples": len(mean_firings),
        }

    mean_firings = np.array(mean_firings)
    valid_depths = np.array(valid_depths)

    # Spearman rank correlation
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(valid_depths, mean_firings)
    except ImportError:
        # Fallback: manual Spearman via rank correlation
        def _rank(x):
            temp = x.argsort().argsort().astype(float)
            return temp
        r_depths = _rank(valid_depths)
        r_firings = _rank(mean_firings)
        n = len(r_depths)
        d = r_depths - r_firings
        r = 1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1))
        p = float("nan")  # approximate not computed

    # Depth-grouped averages
    depth_to_mean = {}
    for d_val in sorted(set(valid_depths)):
        mask = valid_depths == d_val
        depth_to_mean[int(d_val)] = float(mean_firings[mask].mean())

    return {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "depth_to_mean_firing": depth_to_mean,
        "n_examples": len(mean_firings),
    }


def plot_firing_order_vs_depth(
    firing_orders: torch.Tensor,
    proof_depths: torch.Tensor,
    title: str = "Slot Firing Time vs Proof Depth",
    save_path: str | None = None,
):
    """
    Scatter plot + box plot of mean slot firing time vs proof depth.

    This is a key visualization for the paper: if deeper proofs require
    later slot firing, it validates that slots mirror the reasoning chain.

    Args:
        firing_orders: (N, K) tensor, timestep when each slot first fired
        proof_depths: (N,) tensor, proof depth for each example
        title: plot title
        save_path: if provided, save figure
    """
    _check_mpl()

    fo_np = firing_orders.cpu().numpy()
    pd_np = proof_depths.cpu().numpy()

    # Compute mean firing time per example
    mean_firings = []
    depths = []
    for i in range(fo_np.shape[0]):
        fired = fo_np[i][fo_np[i] >= 0]
        if len(fired) > 0:
            mean_firings.append(fired.mean())
            depths.append(pd_np[i])

    if not mean_firings:
        print("  No fired slots to plot.")
        return

    mean_firings = np.array(mean_firings)
    depths = np.array(depths)

    # Compute correlation stats
    stats = compute_firing_depth_correlation(firing_orders, proof_depths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Left: scatter with jitter ───────────────────────────
    jitter = np.random.uniform(-0.15, 0.15, size=len(depths))
    ax1.scatter(depths + jitter, mean_firings, alpha=0.3, s=15, c="#2196F3")

    # Overlay depth-group means
    for d_val, mean_f in stats["depth_to_mean_firing"].items():
        ax1.plot(d_val, mean_f, "ro", markersize=10, zorder=5)

    # Trend line
    if len(set(depths)) > 1:
        z = np.polyfit(depths, mean_firings, 1)
        x_line = np.linspace(depths.min(), depths.max(), 100)
        ax1.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7,
                 label=f"ρ={stats['spearman_r']:.3f}")
        ax1.legend(fontsize=10)

    ax1.set_xlabel("Proof Depth")
    ax1.set_ylabel("Mean Slot Firing Timestep")
    ax1.set_title("Scatter (red = group mean)")
    ax1.grid(True, alpha=0.3)

    # ── Right: box plot by depth ────────────────────────────
    unique_depths = sorted(set(depths))
    box_data = [mean_firings[depths == d] for d in unique_depths]
    bp = ax2.boxplot(box_data, labels=[str(int(d)) for d in unique_depths],
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#E3F2FD")
        patch.set_edgecolor("#1565C0")
    ax2.set_xlabel("Proof Depth")
    ax2.set_ylabel("Mean Slot Firing Timestep")
    ax2.set_title(f"Box Plot (Spearman ρ={stats['spearman_r']:.3f})")
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Firing order plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

    return stats


def plot_per_depth_accuracy(
    depth_results: dict[int, dict],
    title: str = "Per-Depth Accuracy Breakdown",
    save_path: str | None = None,
):
    """
    Bar chart showing accuracy at each proof depth.

    Args:
        depth_results: dict mapping depth → {"accuracy": float, "count": int}
        save_path: if provided, save figure
    """
    _check_mpl()

    depths = sorted(depth_results.keys())
    accs = [depth_results[d]["accuracy"] for d in depths]
    counts = [depth_results[d]["count"] for d in depths]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars = ax1.bar(depths, accs, color="#4CAF50", alpha=0.8, zorder=3)
    ax1.set_xlabel("Proof Depth")
    ax1.set_ylabel("Accuracy", color="#4CAF50")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(title, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)

    for bar, acc, cnt in zip(bars, accs, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{acc:.1%}\n(n={cnt})", ha="center", va="bottom", fontsize=9)

    ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Majority baseline")
    ax1.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Per-depth accuracy chart saved to {save_path}")
    else:
        plt.show()
    plt.close()
