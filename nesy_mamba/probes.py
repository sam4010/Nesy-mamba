"""
Interpretability Probes for NeSy-Mamba.

Three experiment types that produce direct evidence for the paper:

1. **Linear Probe** (Section: "Do slots encode symbolic rules?")
   - Train logistic regression on frozen slot activations → predict which rules fired.
   - Compare against hidden-state baseline (h_final → rules).
   - If slots >> hidden: slots genuinely learned interpretable symbolic structure.

2. **Causal Ablation** (Section: "Are slots causally necessary?")
   - Zero out individual slots and measure answer accuracy drop.
   - Shuffle slot assignments and measure degradation.
   - If accuracy drops: model actually *uses* slot information for predictions.

3. **Compositional Generalisation** (Section: "Does NeSy-Mamba generalise?")
   - Train on depth ≤ D_train, test on depth > D_train.
   - Show performance on unseen proof depths vs baselines.

Usage (after training):
    python -m nesy_mamba.probes --checkpoint path/to/best.pt --data_dir path/to/data

All probes work on CPU with frozen model weights (no GPU needed).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .config import NeSyMambaConfig
from .nesy_mamba import NeSyMamba
from .data_utils import get_dataloaders, SymbolicProofWriterDataset


# ═══════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    model: NeSyMamba,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_examples: int = 0,
) -> dict:
    """
    Extract frozen features from a trained NeSy-Mamba model.

    Returns dict with:
        slot_values:  (N, K)         — final slot activations
        h_final:      (N, d_model)   — last-token hidden state (pre-slot)
        slot_history:  (N, L, K)     — slot values at every timestep
        answer_labels: (N,)          — ground truth answer (0/1)
        slot_labels:   (N, K)        — ground truth slot labels
        answer_probs:  (N,)          — model's predicted answer probability
        proof_depths:  (N,)          — proof depth per example (-1 if unknown)
    """
    model.eval()
    feats = defaultdict(list)
    n_collected = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        B, L = input_ids.shape

        # Forward through embedding + backbone (no dropout in eval)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = model.embedding(input_ids) + model.pos_embedding(positions)
        h_seq = model.backbone(x)                          # (B, L, d_model)

        # Last real token hidden state
        pad_mask = input_ids != 0
        lengths = pad_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, 1, h_seq.size(-1))
        h_final = h_seq.gather(1, idx).squeeze(1)          # (B, d_model)

        # Slot gate
        if model.slot_gate is not None:
            slot_values, slot_history, firing_order = model.slot_gate(h_seq)
        else:
            K = model.cfg.n_slots
            slot_values = torch.zeros(B, K, device=device)
            slot_history = torch.zeros(B, L, K, device=device)

        # Answer head
        if slot_values is not None and model.slot_gate is not None:
            ans_input = torch.cat([h_final, slot_values], dim=-1)
        else:
            ans_input = h_final
        answer_logit = model.answer_head(ans_input).squeeze(-1)
        answer_prob = torch.sigmoid(answer_logit)

        # Collect
        feats["slot_values"].append(slot_values.cpu())
        feats["h_final"].append(h_final.cpu())
        feats["slot_history"].append(slot_history.cpu())
        feats["answer_labels"].append(batch["answer_label"])
        feats["slot_labels"].append(batch["slot_labels"])
        feats["answer_probs"].append(answer_prob.cpu())

        if "proof_depth" in batch:
            feats["proof_depths"].append(batch["proof_depth"])
        else:
            feats["proof_depths"].append(torch.full((B,), -1))

        n_collected += B
        if max_examples > 0 and n_collected >= max_examples:
            break

    # Concatenate all
    result = {}
    for key in feats:
        result[key] = torch.cat(feats[key], dim=0)
        if max_examples > 0:
            result[key] = result[key][:max_examples]

    return result


# ═══════════════════════════════════════════════════════════════════
#  Probe 1: Linear Probe — Slot vs Hidden → Rule Prediction
# ═══════════════════════════════════════════════════════════════════

def linear_probe(
    features: dict,
    n_cv_folds: int = 5,
    max_iter: int = 2000,
) -> dict:
    """
    Train logistic regression probes on two feature sets:
      (a) slot_values (K dims)  → predict each slot_label[k]
      (b) h_final (d_model dims) → predict each slot_label[k]

    For each rule k, we train a binary classifier and compare accuracy.
    If slot_values >> h_final: slots learned interpretable rule features
    that aren't trivially available in the hidden state.

    Requires scikit-learn. Returns empty dict if not installed.

    Returns:
        {
            "per_rule": [
                {"rule": k, "slot_acc": .., "hidden_acc": .., "slot_f1": .., "hidden_f1": ..},
                ...
            ],
            "mean_slot_acc": float,
            "mean_hidden_acc": float,
            "mean_slot_f1": float,
            "mean_hidden_f1": float,
            "slot_wins": int,  # number of rules where slot > hidden
        }
    """
    if not HAS_SKLEARN:
        print("WARNING: scikit-learn not installed. Skipping linear probe.")
        return {"error": "scikit-learn not installed"}
    slots = features["slot_values"].numpy()        # (N, K)
    hidden = features["h_final"].numpy()            # (N, d_model)
    slot_labels = features["slot_labels"].numpy()   # (N, K)

    N, K = slots.shape
    results_per_rule = []

    for k in range(K):
        y = (slot_labels[:, k] > 0.5).astype(int)

        # Skip if all same class (can't probe a constant)
        if y.sum() == 0 or y.sum() == len(y):
            results_per_rule.append({
                "rule": k,
                "slot_acc": float("nan"),
                "hidden_acc": float("nan"),
                "slot_f1": float("nan"),
                "hidden_f1": float("nan"),
                "n_pos": int(y.sum()),
                "skipped": True,
            })
            continue

        # Probe A: slot features → rule k
        clf_slot = LogisticRegression(
            max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        )
        slot_scores = cross_val_score(clf_slot, slots, y, cv=n_cv_folds, scoring="accuracy")
        slot_f1_scores = cross_val_score(clf_slot, slots, y, cv=n_cv_folds, scoring="f1")

        # Probe B: hidden features → rule k
        clf_hidden = LogisticRegression(
            max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        )
        hidden_scores = cross_val_score(clf_hidden, hidden, y, cv=n_cv_folds, scoring="accuracy")
        hidden_f1_scores = cross_val_score(clf_hidden, hidden, y, cv=n_cv_folds, scoring="f1")

        results_per_rule.append({
            "rule": k,
            "slot_acc": float(slot_scores.mean()),
            "hidden_acc": float(hidden_scores.mean()),
            "slot_f1": float(slot_f1_scores.mean()),
            "hidden_f1": float(hidden_f1_scores.mean()),
            "slot_acc_std": float(slot_scores.std()),
            "hidden_acc_std": float(hidden_scores.std()),
            "n_pos": int(y.sum()),
            "skipped": False,
        })

    # Aggregate
    valid = [r for r in results_per_rule if not r.get("skipped", False)]
    mean_slot_acc = np.mean([r["slot_acc"] for r in valid]) if valid else 0.0
    mean_hidden_acc = np.mean([r["hidden_acc"] for r in valid]) if valid else 0.0
    mean_slot_f1 = np.mean([r["slot_f1"] for r in valid]) if valid else 0.0
    mean_hidden_f1 = np.mean([r["hidden_f1"] for r in valid]) if valid else 0.0
    slot_wins = sum(1 for r in valid if r["slot_acc"] > r["hidden_acc"])

    return {
        "per_rule": results_per_rule,
        "mean_slot_acc": float(mean_slot_acc),
        "mean_hidden_acc": float(mean_hidden_acc),
        "mean_slot_f1": float(mean_slot_f1),
        "mean_hidden_f1": float(mean_hidden_f1),
        "slot_wins": slot_wins,
        "total_rules_probed": len(valid),
    }


# ═══════════════════════════════════════════════════════════════════
#  Probe 2: Causal Ablation — Are Slots Necessary?
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def causal_ablation(
    model: NeSyMamba,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_examples: int = 2000,
) -> dict:
    """
    Three causal ablation experiments:

    (a) **Zero-slot**: Set all slot values to 0 before answer head.
        If accuracy drops → model uses slot information.

    (b) **Per-slot zero**: Zero out one slot at a time, measure per-rule impact.
        Shows which slots are most causally important.

    (c) **Shuffle-slot**: Randomly permute slot values across examples in batch.
        If accuracy drops → model relies on *specific* slot-example binding.

    Returns:
        {
            "baseline_acc": float,       # Normal accuracy
            "zero_all_acc": float,       # Accuracy with all slots zeroed
            "shuffle_acc": float,        # Accuracy with shuffled slots
            "per_slot_zero": [           # Per-slot ablation
                {"slot": k, "acc": float, "delta": float}, ...
            ],
            "zero_all_delta": float,     # baseline - zero_all
            "shuffle_delta": float,      # baseline - shuffle
        }
    """
    model.eval()
    if model.slot_gate is None:
        return {"error": "Model has no slot gate (base variant)"}

    K = model.cfg.n_slots

    # Accumulators
    baseline_correct = 0
    zero_all_correct = 0
    shuffle_correct = 0
    per_slot_correct = [0] * K
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        answer_label = batch["answer_label"].to(device)
        B, L = input_ids.shape

        # Forward: embedding + backbone
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = model.embedding(input_ids) + model.pos_embedding(positions)
        h_seq = model.backbone(x)

        # Get hidden final
        pad_mask = input_ids != 0
        lengths = pad_mask.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h_seq.size(-1))
        h_final = h_seq.gather(1, idx).squeeze(1)

        # Normal slot values
        slot_values, _, _ = model.slot_gate(h_seq)

        # (a) Baseline
        ans_input = torch.cat([h_final, slot_values], dim=-1)
        logit = model.answer_head(ans_input).squeeze(-1)
        pred = (torch.sigmoid(logit) > 0.5).float()
        baseline_correct += (pred == answer_label).sum().item()

        # (b) Zero ALL slots
        zero_slots = torch.zeros_like(slot_values)
        ans_input_zero = torch.cat([h_final, zero_slots], dim=-1)
        logit_zero = model.answer_head(ans_input_zero).squeeze(-1)
        pred_zero = (torch.sigmoid(logit_zero) > 0.5).float()
        zero_all_correct += (pred_zero == answer_label).sum().item()

        # (c) Shuffle slots across batch
        perm = torch.randperm(B, device=device)
        shuffled_slots = slot_values[perm]
        ans_input_shuf = torch.cat([h_final, shuffled_slots], dim=-1)
        logit_shuf = model.answer_head(ans_input_shuf).squeeze(-1)
        pred_shuf = (torch.sigmoid(logit_shuf) > 0.5).float()
        shuffle_correct += (pred_shuf == answer_label).sum().item()

        # (d) Per-slot zero
        for k in range(K):
            ablated = slot_values.clone()
            ablated[:, k] = 0.0
            ans_input_k = torch.cat([h_final, ablated], dim=-1)
            logit_k = model.answer_head(ans_input_k).squeeze(-1)
            pred_k = (torch.sigmoid(logit_k) > 0.5).float()
            per_slot_correct[k] += (pred_k == answer_label).sum().item()

        total += B
        if max_examples > 0 and total >= max_examples:
            break

    baseline_acc = baseline_correct / max(total, 1)
    zero_all_acc = zero_all_correct / max(total, 1)
    shuffle_acc = shuffle_correct / max(total, 1)

    per_slot_results = []
    for k in range(K):
        acc_k = per_slot_correct[k] / max(total, 1)
        per_slot_results.append({
            "slot": k,
            "acc": round(acc_k, 4),
            "delta": round(baseline_acc - acc_k, 4),
        })

    return {
        "baseline_acc": round(baseline_acc, 4),
        "zero_all_acc": round(zero_all_acc, 4),
        "shuffle_acc": round(shuffle_acc, 4),
        "zero_all_delta": round(baseline_acc - zero_all_acc, 4),
        "shuffle_delta": round(baseline_acc - shuffle_acc, 4),
        "per_slot_zero": per_slot_results,
    }


# ═══════════════════════════════════════════════════════════════════
#  Probe 3: Compositional Generalisation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compositional_generalisation(
    model: NeSyMamba,
    features: dict,
    device: torch.device,
) -> dict:
    """
    Evaluate model performance split by proof depth.

    Uses pre-extracted features with proof_depths field.
    Reports accuracy per depth and highlights train→test generalisation
    (e.g., if trained on depth ≤ 2, how does it perform on depth 3+).

    Returns:
        {
            "per_depth": {0: {"acc": .., "n": ..}, 1: {...}, ...},
            "overall_acc": float,
        }
    """
    answer_probs = features["answer_probs"]
    answer_labels = features["answer_labels"]
    proof_depths = features["proof_depths"]

    preds = (answer_probs > 0.5).float()
    correct = (preds == answer_labels).float()

    per_depth = {}
    for d in sorted(proof_depths.unique().tolist()):
        d_int = int(d)
        mask = proof_depths == d
        n = mask.sum().item()
        if n == 0:
            continue
        acc = correct[mask].mean().item()
        # Per-class accuracy
        true_mask = mask & (answer_labels > 0.5)
        false_mask = mask & (answer_labels <= 0.5)
        acc_true = correct[true_mask].mean().item() if true_mask.any() else float("nan")
        acc_false = correct[false_mask].mean().item() if false_mask.any() else float("nan")
        per_depth[d_int] = {
            "acc": round(acc, 4),
            "acc_true": round(acc_true, 4),
            "acc_false": round(acc_false, 4),
            "n": int(n),
        }

    overall_acc = correct.mean().item()

    return {
        "per_depth": per_depth,
        "overall_acc": round(overall_acc, 4),
    }


# ═══════════════════════════════════════════════════════════════════
#  Pretty Printing
# ═══════════════════════════════════════════════════════════════════

def print_probe_results(probe_results: dict, rule_names: list[str] | None = None):
    """Print linear probe results in a clean table."""
    print("\n" + "=" * 70)
    print("  LINEAR PROBE: Slot Activations vs Hidden States → Rule Prediction")
    print("=" * 70)

    per_rule = probe_results["per_rule"]
    K = len(per_rule)
    if rule_names is None:
        rule_names = [f"Rule_{i}" for i in range(K)]

    print(f"\n  {'Rule':<25} | {'Slot Acc':>9} | {'Hidden Acc':>10} | {'Gap':>7} | {'Winner':>7}")
    print("  " + "-" * 65)

    for r in per_rule:
        k = r["rule"]
        name = rule_names[k] if k < len(rule_names) else f"Rule_{k}"
        if r.get("skipped"):
            print(f"  {name:<25} | {'SKIP':>9} | {'SKIP':>10} | {'--':>7} | {'--':>7}")
            continue
        gap = r["slot_acc"] - r["hidden_acc"]
        winner = "SLOT" if gap > 0.01 else ("HIDDEN" if gap < -0.01 else "TIE")
        print(f"  {name:<25} | {r['slot_acc']:>8.3f}% | {r['hidden_acc']:>9.3f}% | "
              f"{gap:>+6.3f} | {winner:>7}")

    print("  " + "-" * 65)
    print(f"  {'MEAN':<25} | {probe_results['mean_slot_acc']:>8.3f}% | "
          f"{probe_results['mean_hidden_acc']:>9.3f}% | "
          f"{probe_results['mean_slot_acc'] - probe_results['mean_hidden_acc']:>+6.3f} | "
          f"{'SLOT' if probe_results['slot_wins'] > probe_results['total_rules_probed'] / 2 else 'HIDDEN':>7}")
    print(f"\n  Slot wins: {probe_results['slot_wins']}/{probe_results['total_rules_probed']} rules")


def print_ablation_results(ablation_results: dict, rule_names: list[str] | None = None):
    """Print causal ablation results."""
    print("\n" + "=" * 70)
    print("  CAUSAL ABLATION: Are Slots Necessary for Prediction?")
    print("=" * 70)

    if "error" in ablation_results:
        print(f"  ERROR: {ablation_results['error']}")
        return

    print(f"\n  Baseline accuracy:       {ablation_results['baseline_acc']:.4f}")
    print(f"  Zero ALL slots:          {ablation_results['zero_all_acc']:.4f}  "
          f"(delta = {ablation_results['zero_all_delta']:+.4f})")
    print(f"  Shuffle slots:           {ablation_results['shuffle_acc']:.4f}  "
          f"(delta = {ablation_results['shuffle_delta']:+.4f})")

    K = len(ablation_results["per_slot_zero"])
    if rule_names is None:
        rule_names = [f"Slot_{i}" for i in range(K)]

    print(f"\n  Per-slot zero-out:")
    print(f"  {'Slot':<25} | {'Acc':>7} | {'Delta':>7}")
    print("  " + "-" * 45)
    for r in ablation_results["per_slot_zero"]:
        k = r["slot"]
        name = rule_names[k] if k < len(rule_names) else f"Slot_{k}"
        print(f"  {name:<25} | {r['acc']:>6.4f} | {r['delta']:>+6.4f}")


def print_compgen_results(compgen_results: dict):
    """Print compositional generalisation results."""
    print("\n" + "=" * 70)
    print("  COMPOSITIONAL GENERALISATION: Per-Depth Accuracy")
    print("=" * 70)

    print(f"\n  Overall accuracy: {compgen_results['overall_acc']:.4f}")
    print(f"\n  {'Depth':>6} | {'Acc':>7} | {'Acc_T':>7} | {'Acc_F':>7} | {'N':>7}")
    print("  " + "-" * 45)
    for d, info in sorted(compgen_results["per_depth"].items()):
        print(f"  {d:>6} | {info['acc']:>6.4f} | {info['acc_true']:>6.4f} | "
              f"{info['acc_false']:>6.4f} | {info['n']:>7}")


# ═══════════════════════════════════════════════════════════════════
#  CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NeSy-Mamba Interpretability Probes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best checkpoint (best_full.pt)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="ProofWriter data directory (default: from checkpoint config)")
    parser.add_argument("--max_examples", type=int, default=5000,
                        help="Max examples for probes (0=all)")
    parser.add_argument("--max_depth", type=int, default=5,
                        help="Max proof depth to load")
    parser.add_argument("--probes", type=str, default="all",
                        help="Comma-separated: linear,ablation,compgen,all")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu/cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load checkpoint ─────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfg_dict = ckpt["config"]
    cfg = NeSyMambaConfig.from_dict(cfg_dict)

    if args.data_dir:
        cfg.data_dir = args.data_dir

    model = NeSyMamba(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {cfg.variant}, d_model={cfg.d_model}, n_slots={cfg.n_slots}")

    # ── Load data ───────────────────────────────────────────────
    data_dir = getattr(cfg, "data_dir", None) or args.data_dir
    if data_dir is None:
        print("ERROR: No data_dir in checkpoint config or CLI. Use --data_dir.")
        sys.exit(1)

    print(f"Loading data from: {data_dir}")
    _, val_loader = get_dataloaders(
        dataset_name="proofwriter",
        batch_size=64,
        data_dir=data_dir,
        max_examples=0,
        encoding="symbolic",
        n_slots=cfg.n_slots,
        max_seq_len=cfg.max_seq_len,
        max_depth=args.max_depth,
    )

    rule_names = SymbolicProofWriterDataset.get_rule_names()

    # ── Extract features ────────────────────────────────────────
    print(f"Extracting features (max {args.max_examples} examples)...")
    features = extract_features(model, val_loader, device, max_examples=args.max_examples)
    N = features["slot_values"].shape[0]
    print(f"  Extracted {N} examples")

    # ── Run probes ──────────────────────────────────────────────
    probes_to_run = args.probes.split(",") if args.probes != "all" else ["linear", "ablation", "compgen"]
    all_results = {}

    if "linear" in probes_to_run:
        print("\nRunning linear probe...")
        probe_results = linear_probe(features)
        print_probe_results(probe_results, rule_names)
        all_results["linear_probe"] = probe_results

    if "ablation" in probes_to_run:
        print("\nRunning causal ablation...")
        ablation_results = causal_ablation(model, val_loader, device,
                                           max_examples=args.max_examples)
        print_ablation_results(ablation_results, rule_names)
        all_results["causal_ablation"] = ablation_results

    if "compgen" in probes_to_run:
        print("\nRunning compositional generalisation analysis...")
        compgen_results = compositional_generalisation(model, features, device)
        print_compgen_results(compgen_results)
        all_results["compositional_generalisation"] = compgen_results

    # ── Save results ────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 70)
    print("  ALL PROBES COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()
