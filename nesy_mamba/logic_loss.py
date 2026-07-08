"""
Differentiable Logic Losses for NeSy Mamba.

Implements the three loss components from the slot-gated design:
  1. Slot supervision loss   — push slots to match rule labels
  2. Answer consistency loss — if antecedents fire → answer must be true
  3. Orthogonality loss      — prevent redundant slot representations

Supports both static (global) and dynamic (per-example, proof-conditioned)
rule lists for L_rule computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import NeSyMambaConfig


class LogicLossComputer(nn.Module):
    """
    Computes all differentiable logic losses.

    Args:
        cfg: NeSyMambaConfig with lambda weights.
    """

    def __init__(self, cfg: NeSyMambaConfig):
        super().__init__()
        self.lambda_slot = cfg.lambda_slot
        self.lambda_rule = cfg.lambda_rule
        self.lambda_ortho = cfg.lambda_ortho
        self.lambda_entropy = getattr(cfg, 'lambda_entropy', 0.1)
        # Per-slot positive weighting for class-imbalanced BCE
        # Set via set_slot_pos_weight() after data loading
        self.register_buffer('slot_pos_weight', None)

    def set_slot_pos_weight(self, slot_label_freq: torch.Tensor):
        """
        Compute and store per-slot pos_weight from label frequencies.

        Args:
            slot_label_freq: (K,) tensor of per-slot positive-label
                frequencies in [0, 1].  E.g. slot_label_freq[2] = 0.109
                means slot 2 is active in 10.9% of training examples.

        Sets self.slot_pos_weight = (1 - freq) / freq, capped at 20.
        """
        freq = slot_label_freq.clamp(min=0.01, max=0.99)
        w = (1.0 - freq) / freq
        w = w.clamp(max=20.0)  # cap to prevent extreme gradients
        self.slot_pos_weight = w.to(self.slot_pos_weight.device if self.slot_pos_weight is not None
                                     else freq.device)

    # ── Individual loss terms ───────────────────────────────────────

    def slot_supervision_loss(
        self,
        slot_preds: torch.Tensor,    # (B, K)  predicted slot values ∈ [0,1]
        slot_labels: torch.Tensor,   # (B, K)  ground-truth {0, 1}
    ) -> torch.Tensor:
        """
        L_slot = weighted BCE(slot_preds, slot_labels)

        Per-slot positive weighting to counteract label imbalance.
        Minority rule types (e.g. Rel2Prop at 5%) would otherwise
        be overwhelmed by the 95% negative gradient signal.
        Weight w_k scales the positive (y=1) term:
            L = -[ w_k · y · log(p) + (1-y) · log(1-p) ]
        """
        eps = 1e-6
        preds_safe = slot_preds.clamp(eps, 1.0 - eps)

        if self.slot_pos_weight is not None:
            # Weighted BCE: upweight positive examples for rare slots
            w = self.slot_pos_weight.to(slot_preds.device).unsqueeze(0)  # (1, K)
            loss = -(w * slot_labels * torch.log(preds_safe)
                     + (1.0 - slot_labels) * torch.log(1.0 - preds_safe))
            return loss.mean()
        else:
            return F.binary_cross_entropy(preds_safe, slot_labels)

    @staticmethod
    def answer_consistency_loss(
        slot_preds: torch.Tensor,    # (B, K)
        answer_prob: torch.Tensor,   # (B,)    predicted answer probability
        rules: list[tuple[int, ...]] | None = None,
        per_example_rules: list[list[tuple[int, ...]]] | None = None,
    ) -> torch.Tensor:
        """
        L_rule — enforces logical implication: antecedent slots → answer.

        For single-antecedent rules:
            L = slot_i × (1 − answer_prob)

        For two-antecedent rules:
            L = slot_i × slot_j × (1 − answer_prob)

        Supports two modes:
          1. **Global rules** (rules param): same rule list applied to all
             examples in the batch. Efficient batch-vectorised computation.
          2. **Per-example rules** (per_example_rules param): each example
             has its own rule list extracted from its proof tree.
             per_example_rules[i] is a list of antecedent tuples for
             example i. Takes priority over global rules if both given.

        Args:
            rules: list of tuples, each tuple contains slot indices that
                   form the antecedent of a rule. If None, treats every
                   slot independently (single-antecedent).
            per_example_rules: list of length B, each element is a list of
                   antecedent tuples for that example. Extracted from
                   proof_parser.extract_dynamic_rules().
        """
        B = slot_preds.shape[0]
        neg_ans = (1.0 - answer_prob)  # (B,)

        # ── Per-example dynamic rules (proof-conditioned) ──────────
        if per_example_rules is not None:
            loss = torch.tensor(0.0, device=slot_preds.device, dtype=slot_preds.dtype)
            n_rules_total = 0
            for i in range(B):
                ex_rules = per_example_rules[i]
                if not ex_rules:
                    continue
                for antecedents in ex_rules:
                    # Product of all antecedent slot activations for this example
                    joint = torch.ones(1, device=slot_preds.device, dtype=slot_preds.dtype)
                    for idx in antecedents:
                        if idx < slot_preds.shape[1]:
                            joint = joint * slot_preds[i, idx]
                    loss = loss + joint * neg_ans[i]
                    n_rules_total += 1
            return loss / max(n_rules_total, 1)

        # ── Global rules (same for all examples in batch) ──────────
        if rules is None:
            # Default: each slot is a single-antecedent rule
            # L = mean over slots of: slot_i × (1 − answer)
            return (slot_preds * neg_ans.unsqueeze(1)).mean()

        K = slot_preds.shape[1]
        loss = torch.tensor(0.0, device=slot_preds.device, dtype=slot_preds.dtype)
        n_valid = 0
        for antecedents in rules:
            # Skip rules with out-of-bounds slot indices
            if any(idx >= K for idx in antecedents):
                continue
            # Product of all antecedent slot activations
            joint = torch.ones(B, device=slot_preds.device, dtype=slot_preds.dtype)
            for idx in antecedents:
                joint = joint * slot_preds[:, idx]
            loss = loss + (joint * neg_ans).mean()
            n_valid += 1

        return loss / max(n_valid, 1)

    @staticmethod
    def orthogonality_loss(
        slot_preds: torch.Tensor,    # (B, K)
    ) -> torch.Tensor:
        """
        L_ortho = ‖S^T · S − I_K‖_F

        Encourages each slot's activation pattern to be distinct and
        orthonormal across the batch, preventing two slots from
        learning the same predicate.
        """
        K = slot_preds.shape[1]
        # L2-normalise each slot column across the batch
        S = slot_preds                              # (B, K)
        S_norm = S / (S.norm(dim=0, keepdim=True) + 1e-8)  # (B, K)
        gram = S_norm.T @ S_norm                    # (K, K) ≈ cosine sim
        eye = torch.eye(K, device=S.device, dtype=S.dtype)
        return torch.norm(gram - eye, p="fro")

    @staticmethod
    def slot_entropy_loss(
        slot_preds: torch.Tensor,    # (B, K)
    ) -> torch.Tensor:
        """
        L_entropy = -mean[ H(s_i) ]  (negated so that maximising entropy
        means *minimising* this loss).

        For each slot value s_i ∈ (0, 1), the Bernoulli entropy is:
            H(s) = -s log(s) - (1-s) log(1-s)

        We want slots to be decisive (near 0 or 1) for active use,
        BUT we don't want ALL slots to collapse to the same value.
        So we penalise low variance across slots (anti-collapse):
            L = -std(slot_means_across_examples_per_slot)

        If all K slots have the same mean activation, std ≈ 0 → high loss.
        If slots are differentiated, std > 0 → low loss.
        """
        # Per-slot mean activation across the batch
        slot_means = slot_preds.mean(dim=0)  # (K,)
        # We want diversity: penalise low std across slot means
        # Negative std = we minimise this loss by maximising diversity
        diversity = slot_means.std()
        # Also penalise if ALL slots are near 1.0 (collapsed-high)
        # or ALL near 0.0 (collapsed-low) via distance from 0.5 mean
        mean_activation = slot_means.mean()
        collapse_penalty = (mean_activation - 0.5).pow(2)
        return -diversity + collapse_penalty

    # ── Combined loss ───────────────────────────────────────────────

    def forward(
        self,
        slot_preds: torch.Tensor,           # (B, K)
        slot_labels: torch.Tensor | None,   # (B, K) or None
        answer_prob: torch.Tensor | None,   # (B,) or None
        rules: list[tuple[int, ...]] | None = None,
        per_example_rules: list[list[tuple[int, ...]]] | None = None,
        enable_rule_loss: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute total logic loss = λ₁·L_slot + λ₂·L_rule + λ₃·L_ortho.

        Args:
            per_example_rules: If provided, overrides global rules for L_rule.
                Each element is a list of antecedent tuples for one example.

        Returns:
            total_loss: scalar tensor
            breakdown: dict with individual loss values for logging
        """
        device = slot_preds.device
        dtype = slot_preds.dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)
        breakdown = {}

        # Slot supervision (with per-slot class-imbalance weighting)
        if slot_labels is not None:
            l_slot = self.slot_supervision_loss(slot_preds, slot_labels)
            total = total + self.lambda_slot * l_slot
            breakdown["L_slot"] = l_slot.item()

        # Answer consistency (gated by warmup schedule)
        if enable_rule_loss and answer_prob is not None:
            l_rule = self.answer_consistency_loss(
                slot_preds, answer_prob, rules, per_example_rules
            )
            total = total + self.lambda_rule * l_rule
            breakdown["L_rule"] = l_rule.item()

        # Orthogonality
        l_ortho = self.orthogonality_loss(slot_preds)
        total = total + self.lambda_ortho * l_ortho
        breakdown["L_ortho"] = l_ortho.item()

        # Slot entropy / anti-collapse
        if self.lambda_entropy > 0:
            l_entropy = self.slot_entropy_loss(slot_preds)
            total = total + self.lambda_entropy * l_entropy
            breakdown["L_entropy"] = l_entropy.item()

        breakdown["L_logic_total"] = total.item()
        return total, breakdown
