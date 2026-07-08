"""
Evaluation Metrics for Slot-Gated NeSy Mamba.

Includes:
  - Accuracy (standard)
  - Logic Fidelity: % examples satisfying all declared rules
  - Slot Fidelity: agreement between slot activations and ground truth
  - AUC-MoRF: perturbation-based explanation quality metric
"""

import torch
import numpy as np


def accuracy(
    answer_prob: torch.Tensor,   # (B,)
    answer_label: torch.Tensor,  # (B,)
    threshold: float = 0.5,
) -> float:
    """Standard binary classification accuracy."""
    preds = (answer_prob > threshold).long()
    return (preds == answer_label.long()).float().mean().item()


def logic_fidelity(
    slot_preds: torch.Tensor,             # (B, K)
    answer_prob: torch.Tensor,            # (B,)
    rules: list[tuple[int, ...]] | None,  # antecedent slot indices
    threshold: float = 0.5,
) -> float:
    """
    Logic Fidelity: fraction of examples where all declared rules
    are logically satisfied.

    A rule (antecedent_slots → answer) is satisfied when:
        - If all antecedent slots fire (>threshold), then answer is True (>threshold)
        - OR at least one antecedent slot doesn't fire
    """
    B = slot_preds.shape[0]
    if rules is None:
        return 1.0  # no rules to violate

    slot_fired = slot_preds > threshold    # (B, K) bool
    ans_true = answer_prob > threshold     # (B,) bool

    all_satisfied = torch.ones(B, dtype=torch.bool, device=slot_preds.device)

    for antecedents in rules:
        # Check if all antecedents fire
        ante_all_fire = torch.ones(B, dtype=torch.bool, device=slot_preds.device)
        for idx in antecedents:
            ante_all_fire = ante_all_fire & slot_fired[:, idx]

        # Rule satisfied if: ¬(all antecedents fire) OR answer is true
        rule_ok = (~ante_all_fire) | ans_true
        all_satisfied = all_satisfied & rule_ok

    return all_satisfied.float().mean().item()


def slot_fidelity(
    slot_preds: torch.Tensor,    # (B, K)
    slot_labels: torch.Tensor,   # (B, K)
    threshold: float = 0.5,
) -> float:
    """
    Slot Fidelity: fraction of (example, slot) pairs where the
    slot activation correctly matches the ground truth.
    """
    pred_binary = (slot_preds > threshold).float()
    return (pred_binary == slot_labels.float()).float().mean().item()


def auc_morf(
    model,
    x: torch.Tensor,             # (B, L)  input token IDs
    slot_values: torch.Tensor,    # (B, K)
    answer_prob_orig: torch.Tensor,  # (B,)
    n_steps: int = 5,
    attribution: torch.Tensor | None = None,  # (B, L) token importance
) -> float:
    """
    AUC-MoRF (Most Relevant First).

    Iteratively mask input positions starting from the most important,
    measure confidence drop. Lower AUC = better explanation (removing
    important inputs hurts more).

    If `attribution` is provided (from model.slot_token_attribution()),
    tokens are masked in order of their attribution scores (best method).
    Otherwise falls back to heuristic positional masking.

    Args:
        model: NeSyMamba model
        x: (B, L) input token IDs
        slot_values: (B, K) slot activations
        answer_prob_orig: (B,) original model confidence
        n_steps: number of progressive masking steps
        attribution: (B, L) importance scores per token (optional)
    """
    B, L = x.shape
    device = x.device

    confidences = [answer_prob_orig.mean().item()]
    x_masked = x.clone()

    # Number of tokens to mask per step
    mask_per_step = max(1, L // (n_steps + 1))

    if attribution is not None:
        # ── Attribution-guided masking (proper MoRF) ────────────
        # Sort positions by importance (descending) for each example
        _, sorted_positions = attribution.sort(dim=-1, descending=True)  # (B, L)
        already_masked = 0

        for step in range(n_steps):
            # Mask the next batch of most important positions
            end_pos = already_masked + mask_per_step
            for b in range(B):
                positions_to_mask = sorted_positions[b, already_masked:end_pos]
                x_masked[b, positions_to_mask] = 0  # PAD token
            already_masked = end_pos

            with torch.no_grad():
                model.eval()
                result = model(x_masked)
                conf = result["answer_prob"].mean().item()
            confidences.append(conf)
    else:
        # ── Fallback: heuristic positional masking ──────────────
        for step in range(n_steps):
            start = step * mask_per_step
            end = min(start + mask_per_step, L)
            x_masked[:, start:end] = 0
            with torch.no_grad():
                model.eval()
                result = model(x_masked)
                conf = result["answer_prob"].mean().item()
            confidences.append(conf)

    # Compute AUC via trapezoidal rule
    fractions = np.linspace(0, 1, len(confidences))
    auc = float(np.trapz(confidences, fractions))
    return auc


def auc_morf_with_attribution(
    model,
    x: torch.Tensor,             # (B, L)
    n_steps: int = 5,
    n_ig_steps: int = 20,
) -> dict[str, float]:
    """
    Convenience function: compute AUC-MoRF using integrated-gradient
    attribution from each slot, then return per-slot scores.

    Returns dict with:
        - "auc_morf_slot_{i}": AUC for slot i
        - "auc_morf_mean": mean across slots
        - "auc_morf_random": AUC with no attribution (baseline)
    """
    with torch.no_grad():
        result = model(x)
    answer_prob = result["answer_prob"]
    slot_values = result["slot_values"]

    results = {}

    # Random baseline (no attribution)
    results["auc_morf_random"] = auc_morf(
        model, x, slot_values, answer_prob, n_steps
    )

    # Per-slot attribution
    if hasattr(model, "slot_token_attribution") and slot_values is not None:
        K = slot_values.shape[1]
        slot_aucs = []
        for k in range(K):
            attr = model.slot_token_attribution(x, slot_idx=k, n_steps=n_ig_steps)
            auc_k = auc_morf(model, x, slot_values, answer_prob, n_steps, attr)
            results[f"auc_morf_slot_{k}"] = auc_k
            slot_aucs.append(auc_k)
        results["auc_morf_mean"] = float(np.mean(slot_aucs))

    return results


def compute_metrics(
    answer_prob: torch.Tensor,
    answer_label: torch.Tensor,
    slot_preds: torch.Tensor,
    slot_labels: torch.Tensor | None,
    rules: list[tuple[int, ...]] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all metrics and return as a dict."""
    metrics = {
        "accuracy": accuracy(answer_prob, answer_label, threshold),
        "logic_fidelity": logic_fidelity(slot_preds, answer_prob, rules, threshold),
    }
    if slot_labels is not None:
        metrics["slot_fidelity"] = slot_fidelity(
            slot_preds, slot_labels, threshold
        )
    return metrics
