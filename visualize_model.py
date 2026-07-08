"""
Visualize NeSy-Mamba model outputs from the λ=0.3 checkpoint.

Generates:
  1. Slot activation heatmap (s_k(t) over tokens) for selected examples
  2. Per-slot activation distribution across validation set
  3. α_k(t) dynamic memory gate trace for selected examples

All figures saved as PDFs in paper/ directory.
"""

import sys, os, json, re
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ── Setup paths ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from nesy_mamba.config import NeSyMambaConfig
from nesy_mamba.nesy_mamba import NeSyMamba
from nesy_mamba.data_utils import (
    SymbolicProofWriterDataset, SimpleVocab, collate_fn, RULE_TYPE_NAMES,
)

CKPT_PATH  = os.path.join(ROOT, "downloads", "lam03", "best_lam03.pt")
DATA_DIR   = os.path.join(ROOT, "nesy_mamba", "data")
OUT_DIR    = os.path.join(ROOT, "paper")
os.makedirs(OUT_DIR, exist_ok=True)

SLOT_NAMES = [
    "PropImpl",   # 0
    "PropConj",   # 1
    "RelChain",   # 2
    "Prop2Rel",   # 3
    "Rel2Prop",   # 4
    "Mix→Rel",    # 5
    "Mix→Prop",   # 6
]

# Consistent color palette for 7 slots
SLOT_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
]


def load_model(ckpt_path):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = NeSyMambaConfig.from_dict(ckpt["config"])
    print(f"  Config: d_model={cfg.d_model}, n_slots={cfg.n_slots}, "
          f"mode={cfg.slot_gate_mode}, variant={cfg.variant}")
    print(f"  Epoch: {ckpt['epoch']}, Val acc: {ckpt['val_metrics']['accuracy']:.4f}")

    model = NeSyMamba(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    return model, cfg


def load_val_data(cfg):
    """Load validation dataset."""
    print(f"\nLoading validation data from {DATA_DIR}...")
    vocab = SimpleVocab()
    train_ds = SymbolicProofWriterDataset(
        os.path.join(DATA_DIR, "proofwriter"),
        split="train", vocab=vocab,
        max_seq_len=cfg.max_seq_len, n_slots=cfg.n_slots,
        max_depth=5,
    )
    vocab.freeze()
    val_ds = SymbolicProofWriterDataset(
        os.path.join(DATA_DIR, "proofwriter"),
        split="val", vocab=vocab,
        max_seq_len=cfg.max_seq_len, n_slots=cfg.n_slots,
        max_depth=5,
    )
    return val_ds, vocab


@torch.no_grad()
def run_inference(model, examples, vocab=None):
    """Run inference on a batch of examples, extracting slot_history and alpha."""
    batch = collate_fn(examples)
    input_ids = batch["input_ids"]
    
    # Forward pass
    out = model(input_ids=input_ids)
    
    # Also extract alpha trace by hooking into the slot gate
    # We'll re-run the slot gate manually to get alpha
    h_seq = model.backbone(
        model.embed_drop(
            model.embedding(input_ids)
            + model.pos_embedding(
                torch.arange(input_ids.shape[1]).unsqueeze(0).expand(input_ids.shape[0], -1)
            )
        )
    )
    
    alpha_history = extract_alpha_trace(model, h_seq)
    
    return {
        "answer_prob": out["answer_prob"],
        "slot_values": out["slot_values"],
        "slot_history": out["slot_history"],
        "firing_order": out["firing_order"],
        "alpha_history": alpha_history,
        "input_ids": input_ids,
        "answer_labels": batch["answer_label"],
        "slot_labels": batch["slot_labels"],
        "proof_depths": batch.get("proof_depth", None),
    }


@torch.no_grad()
def extract_alpha_trace(model, h_seq):
    """Extract α_k(t) trace from the EMA slot gate."""
    if model.slot_gate is None:
        return None
    sg = model.slot_gate
    if sg.gate_mode != "ema" or not sg.dynamic_alpha:
        return None
    
    B, L, _ = h_seq.shape
    alphas = []
    for t in range(L):
        h_t = h_seq[:, t, :]
        alpha_t = torch.sigmoid(sg.W_alpha(h_t))  # (B, K)
        alphas.append(alpha_t)
    return torch.stack(alphas, dim=1)  # (B, L, K)


def get_token_labels(input_ids, vocab):
    """Convert input IDs back to token strings for axis labels."""
    idx2word = vocab.idx2word
    labels = []
    for tid in input_ids:
        tid = tid.item()
        word = idx2word.get(tid, f"?{tid}")
        if tid == 0:
            word = ""  # PAD
        labels.append(word)
    return labels


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Slot Activation Heatmap for Individual Examples
# ═══════════════════════════════════════════════════════════════════

def plot_slot_heatmap(results, example_indices, vocab, save_path):
    """
    Plot slot activation trajectories s_k(t) as heatmaps.
    Shows how each slot evolves across the token sequence.
    """
    n_examples = len(example_indices)
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3.0 * n_examples),
                              constrained_layout=True)
    if n_examples == 1:
        axes = [axes]
    
    # Custom colormap: white → slot color (per-slot) or unified blue
    cmap = LinearSegmentedColormap.from_list("slot", ["#ffffff", "#1b4f72", "#154360"])
    
    for row, idx in enumerate(example_indices):
        ax = axes[row]
        slot_hist = results["slot_history"][idx].numpy()  # (L, K)
        input_ids = results["input_ids"][idx]
        answer_label = results["answer_labels"][idx].item()
        answer_pred = results["answer_prob"][idx].item()
        depth = results["proof_depths"][idx].item() if results["proof_depths"] is not None else "?"
        
        # Find actual sequence length (non-PAD)
        seq_len = (input_ids != 0).sum().item()
        slot_hist = slot_hist[:seq_len, :]  # trim padding
        
        # Transpose: slots on y-axis, tokens on x-axis
        im = ax.imshow(slot_hist.T, aspect="auto", cmap="YlOrRd",
                        vmin=0, vmax=1, interpolation="nearest")
        
        ax.set_yticks(range(7))
        ax.set_yticklabels(SLOT_NAMES, fontsize=9)
        
        # Token labels on x-axis (show every Nth)
        token_labels = get_token_labels(input_ids[:seq_len], vocab)
        n_ticks = min(20, seq_len)
        tick_positions = np.linspace(0, seq_len - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([token_labels[i] for i in tick_positions],
                           rotation=45, ha="right", fontsize=7)
        
        correct = "✓" if (answer_pred > 0.5) == (answer_label > 0.5) else "✗"
        ax.set_title(
            f"Example {idx}: depth={depth}, "
            f"label={'T' if answer_label > 0.5 else 'F'}, "
            f"pred={answer_pred:.3f} {correct}",
            fontsize=10, fontweight="bold",
        )
    
    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Slot Activation $s_k(t)$")
    
    fig.suptitle("Slot Activation Trajectories Across Token Sequence",
                 fontsize=13, fontweight="bold", y=1.02)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 2: α_k(t) Dynamic Memory Gate Trace
# ═══════════════════════════════════════════════════════════════════

def plot_alpha_trace(results, example_idx, vocab, save_path):
    """
    Plot α_k(t) — the dynamic EMA memory coefficient — for one example.
    Shows how the model dynamically controls slot persistence vs. update.
    """
    if results["alpha_history"] is None:
        print("  Skipping alpha trace (not EMA dynamic mode)")
        return
    
    alpha_hist = results["alpha_history"][example_idx].numpy()  # (L, K)
    input_ids = results["input_ids"][example_idx]
    seq_len = (input_ids != 0).sum().item()
    alpha_hist = alpha_hist[:seq_len, :]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [2, 1.5]},
                                     constrained_layout=True)
    
    # Top: α heatmap
    im = ax1.imshow(alpha_hist.T, aspect="auto", cmap="coolwarm",
                     vmin=0.5, vmax=1.0, interpolation="nearest")
    ax1.set_yticks(range(7))
    ax1.set_yticklabels(SLOT_NAMES, fontsize=9)
    ax1.set_xlabel("")
    ax1.set_title(r"Dynamic Memory Gate $\alpha_k(t)$" + f"  (Example {example_idx})",
                  fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax1, shrink=0.8, label=r"$\alpha_k(t)$")
    
    # Bottom: α line traces
    t_axis = np.arange(seq_len)
    for k in range(7):
        ax2.plot(t_axis, alpha_hist[:, k], color=SLOT_COLORS[k],
                 linewidth=1.5, alpha=0.85, label=SLOT_NAMES[k])
    ax2.set_ylabel(r"$\alpha_k(t)$", fontsize=10)
    ax2.set_xlabel("Token position", fontsize=10)
    ax2.set_ylim(0.5, 1.0)
    ax2.legend(ncol=4, fontsize=8, loc="lower left")
    ax2.set_title("Per-Slot Memory Coefficients Over Sequence", fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Token labels
    token_labels = get_token_labels(input_ids[:seq_len], vocab)
    n_ticks = min(20, seq_len)
    tick_positions = np.linspace(0, seq_len - 1, n_ticks, dtype=int)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([token_labels[i] for i in tick_positions],
                        rotation=45, ha="right", fontsize=7)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([token_labels[i] for i in tick_positions],
                        rotation=45, ha="right", fontsize=7)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Per-Slot Activation Distribution (violin plot)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_slot_stats(model, val_ds, max_examples=2000, batch_size=64):
    """Collect slot_values across the val set for distribution plot."""
    from torch.utils.data import DataLoader
    
    n = min(len(val_ds), max_examples)
    indices = list(range(n))
    subset = [val_ds[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)
    
    all_slots = []
    all_depths = []
    all_correct = []
    
    for batch in loader:
        out = model(input_ids=batch["input_ids"])
        sv = out["slot_values"]  # (B, K)
        pred = (out["answer_prob"] > 0.5).float()
        correct = (pred == batch["answer_label"]).float()
        all_slots.append(sv.numpy())
        if "proof_depth" in batch:
            all_depths.append(batch["proof_depth"].numpy())
        all_correct.append(correct.numpy())
    
    all_slots = np.concatenate(all_slots, axis=0)     # (N, K)
    all_correct = np.concatenate(all_correct, axis=0)  # (N,)
    all_depths = np.concatenate(all_depths, axis=0) if all_depths else None
    
    return all_slots, all_correct, all_depths


def plot_slot_distributions(all_slots, all_correct, save_path):
    """
    Violin plot of per-slot activation distributions across the val set.
    Split by correct/incorrect predictions to show slot–accuracy relationship.
    """
    K = all_slots.shape[1]
    
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    
    positions = np.arange(K)
    width = 0.35
    
    # Correct predictions
    correct_mask = all_correct > 0.5
    incorrect_mask = ~correct_mask
    
    data_correct = [all_slots[correct_mask, k] for k in range(K)]
    data_incorrect = [all_slots[incorrect_mask, k] for k in range(K)]
    
    # Violin: correct (left) vs incorrect (right)
    vp1 = ax.violinplot(data_correct, positions=positions - width/2,
                        widths=width, showmeans=True, showmedians=False)
    vp2 = ax.violinplot(data_incorrect, positions=positions + width/2,
                        widths=width, showmeans=True, showmedians=False)
    
    # Color the violins
    for body in vp1["bodies"]:
        body.set_facecolor("#4daf4a")
        body.set_alpha(0.7)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp1:
            vp1[part].set_color("#2d6a2d")
    
    for body in vp2["bodies"]:
        body.set_facecolor("#e41a1c")
        body.set_alpha(0.7)
    for part in ["cmeans", "cmins", "cmaxes", "cbars"]:
        if part in vp2:
            vp2[part].set_color("#a01010")
    
    ax.set_xticks(positions)
    ax.set_xticklabels(SLOT_NAMES, fontsize=9)
    ax.set_ylabel("Slot Activation Value", fontsize=11)
    ax.set_xlabel("Slot (Rule Type)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Per-Slot Activation Distribution: Correct vs. Incorrect Predictions",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    
    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#4daf4a", alpha=0.7, label=f"Correct (n={correct_mask.sum()})"),
        Patch(facecolor="#e41a1c", alpha=0.7, label=f"Incorrect (n={incorrect_mask.sum()})"),
    ], loc="upper right", fontsize=9)
    
    # Annotate means
    for k in range(K):
        mc = data_correct[k].mean()
        mi = data_incorrect[k].mean()
        ax.text(k - width/2, mc + 0.03, f"{mc:.2f}", ha="center", fontsize=7, color="#2d6a2d")
        ax.text(k + width/2, mi + 0.03, f"{mi:.2f}", ha="center", fontsize=7, color="#a01010")
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Slot Activation by Proof Depth
# ═══════════════════════════════════════════════════════════════════

def plot_slots_by_depth(all_slots, all_depths, save_path):
    """
    Grouped bar chart: mean slot activation per proof depth.
    Shows how different rule types activate at different reasoning depths.
    """
    if all_depths is None:
        print("  Skipping depth plot (no depth info)")
        return
    
    K = all_slots.shape[1]
    depths = sorted(set(all_depths.tolist()))
    
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    
    x = np.arange(len(depths))
    total_width = 0.8
    bar_width = total_width / K
    
    for k in range(K):
        means = []
        stds = []
        for d in depths:
            mask = all_depths == d
            vals = all_slots[mask, k]
            means.append(vals.mean())
            stds.append(vals.std() / np.sqrt(len(vals)))  # SEM
        
        offset = (k - K / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.9,
                      yerr=stds, capsize=2,
                      color=SLOT_COLORS[k], alpha=0.85,
                      label=SLOT_NAMES[k],
                      edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Depth {int(d)}" for d in depths], fontsize=10)
    ax.set_ylabel("Mean Slot Activation", fontsize=11)
    ax.set_xlabel("Proof Depth", fontsize=11)
    ax.set_title("Slot Activation by Proof Depth ($\\lambda=0.3$)",
                 fontsize=12, fontweight="bold")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, None)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Slot trajectory line plot (s_k(t) per slot, single example)
# ═══════════════════════════════════════════════════════════════════

def plot_slot_trajectory_lines(results, example_idx, vocab, save_path):
    """
    Line plot of s_k(t) for all 7 slots on one example.
    Emphasises the temporal dynamics of slot activation.
    """
    slot_hist = results["slot_history"][example_idx].numpy()  # (L, K)
    input_ids = results["input_ids"][example_idx]
    seq_len = (input_ids != 0).sum().item()
    slot_hist = slot_hist[:seq_len, :]
    
    answer_label = results["answer_labels"][example_idx].item()
    answer_pred = results["answer_prob"][example_idx].item()
    depth = results["proof_depths"][example_idx].item() if results["proof_depths"] is not None else "?"
    
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    
    t_axis = np.arange(seq_len)
    for k in range(7):
        ax.plot(t_axis, slot_hist[:, k], color=SLOT_COLORS[k],
                linewidth=2.0, alpha=0.85, label=SLOT_NAMES[k])
    
    # Mark special tokens
    token_labels = get_token_labels(input_ids[:seq_len], vocab)
    for i, lbl in enumerate(token_labels):
        if lbl in ("[SEP]", "[FACT]", "[RULE]", "[IMP]"):
            ax.axvline(i, color="gray", alpha=0.2, linewidth=0.8, linestyle="--")
    
    # Find [SEP] position
    sep_pos = None
    for i, lbl in enumerate(token_labels):
        if lbl == "[SEP]":
            sep_pos = i
            break
    if sep_pos is not None:
        ax.axvline(sep_pos, color="black", alpha=0.5, linewidth=1.5, linestyle="-")
        ax.text(sep_pos + 0.5, 0.95, "[SEP]", fontsize=8,
                color="black", alpha=0.7, va="top")
        ax.text(sep_pos / 2, -0.08, "← Query", ha="center", fontsize=8,
                color="#555", transform=ax.get_xaxis_transform())
        ax.text((sep_pos + seq_len) / 2, -0.08, "Facts + Rules →", ha="center",
                fontsize=8, color="#555", transform=ax.get_xaxis_transform())
    
    ax.axhline(0.5, color="gray", linewidth=1, linestyle=":", alpha=0.5, label="threshold")
    ax.set_ylabel(r"Slot Value $s_k(t)$", fontsize=11)
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, seq_len - 1)
    
    correct = "✓" if (answer_pred > 0.5) == (answer_label > 0.5) else "✗"
    ax.set_title(
        f"Slot Trajectories — depth={depth}, "
        f"label={'True' if answer_label > 0.5 else 'False'}, "
        f"pred={answer_pred:.3f} {correct}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NeSy-Mamba Model Visualization")
    print("=" * 60)
    
    # 1) Load model
    model, cfg = load_model(CKPT_PATH)
    
    # 2) Load validation data
    val_ds, vocab = load_val_data(cfg)
    
    # 3) Select interesting examples (diverse depths, mix of T/F)
    print("\nSelecting examples for visualization...")
    
    # Find examples at different depths
    depth_examples = {}
    for i in range(min(len(val_ds), 5000)):
        d = val_ds.data[i]["proof_depth"]
        label = val_ds.data[i]["answer_label"].item()
        key = (d, int(label))
        if key not in depth_examples:
            depth_examples[key] = i
    
    # Pick: depth 0/True, depth 2/True, depth 4/True, depth 3/False
    target_picks = [(0, 1), (2, 1), (4, 1), (3, 0)]
    selected = []
    for target in target_picks:
        if target in depth_examples:
            selected.append(depth_examples[target])
    
    # Fallback: just use first few
    if len(selected) < 3:
        selected = [0, 100, 500, 1000]
    
    print(f"  Selected examples: {selected}")
    
    # 4) Run inference on selected examples
    print("\nRunning inference on selected examples...")
    examples = [val_ds[i] for i in selected]
    results = run_inference(model, examples, vocab)
    
    # Print summary
    for i, idx in enumerate(selected):
        sv = results["slot_values"][i]
        prob = results["answer_prob"][i].item()
        label = results["answer_labels"][i].item()
        depth = results["proof_depths"][i].item() if results["proof_depths"] is not None else "?"
        alive = (sv > 0.1).sum().item()
        print(f"  Ex {idx}: depth={depth}, label={'T' if label > 0.5 else 'F'}, "
              f"pred={prob:.3f}, alive_slots={alive}/7")
        active = [(SLOT_NAMES[k], f"{sv[k]:.3f}") for k in range(7) if sv[k] > 0.1]
        if active:
            print(f"    Active: {active}")
    
    # 5) Generate figures
    print("\n" + "=" * 60)
    print("  Generating Figures")
    print("=" * 60)
    
    # Fig A: Slot activation heatmaps (all selected examples)
    print("\n[Fig A] Slot activation heatmaps...")
    plot_slot_heatmap(
        results, list(range(len(selected))), vocab,
        os.path.join(OUT_DIR, "fig_slot_heatmap.pdf"),
    )
    
    # Fig B: Slot trajectory lines (best example — pick deepest correct)
    print("\n[Fig B] Slot trajectory line plot...")
    # Find the deepest correctly-predicted example
    best_ex = 0
    best_depth = -1
    for i in range(len(selected)):
        prob = results["answer_prob"][i].item()
        label = results["answer_labels"][i].item()
        depth = results["proof_depths"][i].item() if results["proof_depths"] is not None else 0
        if (prob > 0.5) == (label > 0.5) and depth > best_depth:
            best_depth = depth
            best_ex = i
    plot_slot_trajectory_lines(
        results, best_ex, vocab,
        os.path.join(OUT_DIR, "fig_slot_trajectory.pdf"),
    )
    
    # Fig C: Alpha memory gate trace
    print("\n[Fig C] Alpha memory gate trace...")
    plot_alpha_trace(
        results, best_ex, vocab,
        os.path.join(OUT_DIR, "fig_alpha_trace.pdf"),
    )
    
    # 6) Collect val-set statistics for distribution plots
    print("\n[Fig D] Collecting val-set slot statistics (up to 2000 examples)...")
    all_slots, all_correct, all_depths = collect_slot_stats(
        model, val_ds, max_examples=2000, batch_size=64,
    )
    print(f"  Collected {all_slots.shape[0]} examples, "
          f"accuracy = {all_correct.mean():.4f}")
    print(f"  Slot means: {all_slots.mean(axis=0).round(3)}")
    print(f"  Slot stds:  {all_slots.std(axis=0).round(3)}")
    
    # Fig D: Slot activation distributions
    print("\n[Fig D] Slot activation distribution (violin plot)...")
    plot_slot_distributions(
        all_slots, all_correct,
        os.path.join(OUT_DIR, "fig_slot_distributions.pdf"),
    )
    
    # Fig E: Slot activation by depth
    print("\n[Fig E] Slot activation by proof depth...")
    plot_slots_by_depth(
        all_slots, all_depths,
        os.path.join(OUT_DIR, "fig_slots_by_depth.pdf"),
    )
    
    print("\n" + "=" * 60)
    print("  Done! Figures saved to paper/")
    print("=" * 60)


if __name__ == "__main__":
    main()
