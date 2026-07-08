"""
NeSyMamba — End-to-End Slot-Gated Neuro-Symbolic Mamba Model.

Glues together:
  Token Embedding → Mamba Backbone → Slot Gate → Answer Head → Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import NeSyMambaConfig
from .mamba_block import MambaBackbone
from .slot_gate import SlotGate
from .logic_loss import LogicLossComputer


class NeSyMamba(nn.Module):
    """
    Self-explaining Mamba model with symbolic slots and logic losses.

    Supports 4 ablation variants via cfg.variant:
        "base"       — vanilla Mamba (no slots, no logic loss)
        "slots_only" — Mamba + slots (no logic loss)
        "loss_only"  — Mamba + logic loss on outputs (no explicit slots)
        "full"       — Mamba + slots + logic losses
    """

    def __init__(self, cfg: NeSyMambaConfig):
        super().__init__()
        self.cfg = cfg

        # ── Token embedding ─────────────────────────────────────────
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.embed_drop = nn.Dropout(0.1)

        # ── Mamba backbone ──────────────────────────────────────────
        self.backbone = MambaBackbone(cfg)

        # ── Symbolic slots (if enabled) ─────────────────────────────
        if cfg.use_slots:
            self.slot_gate = SlotGate(cfg)
        else:
            self.slot_gate = None

        # ── Answer head ─────────────────────────────────────────────
        # Input: h_final (+ s_final if slots enabled)
        ans_input_dim = cfg.d_model + (cfg.n_slots if cfg.use_slots else 0)
        self.answer_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(ans_input_dim, cfg.d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model, 1),
        )

        # ── Logic loss computer ─────────────────────────────────────
        if cfg.use_logic_loss:
            self.logic_loss = LogicLossComputer(cfg)
        else:
            self.logic_loss = None

        # ── Task loss (with optional class balancing) ───────────────
        # pos_weight > 1 upweights the positive (True) class
        self.register_buffer(
            "_pos_weight",
            torch.tensor([getattr(cfg, "pos_weight", 1.0)]),
        )
        self.task_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight,
        )

    def set_slot_pos_weight(self, slot_label_freq: torch.Tensor):
        """Forward to logic loss computer to set per-slot BCE weights."""
        if self.logic_loss is not None:
            self.logic_loss.set_slot_pos_weight(slot_label_freq)

    def forward(
        self,
        input_ids: torch.Tensor,                     # (B, L) long
        answer_labels: torch.Tensor | None = None,   # (B,) float {0, 1}
        slot_labels: torch.Tensor | None = None,     # (B, K) float {0, 1}
        rules: list[tuple[int, ...]] | None = None,  # antecedent definitions
        per_example_rules: list[list[tuple[int, ...]]] | None = None,
        current_epoch: int = 0,
    ) -> dict:
        """
        Forward pass.

        Returns dict with keys:
            answer_prob:   (B,)     — predicted answer probability
            slot_values:   (B, K)   — final slot activations (or None)
            slot_history:  (B,L,K)  — slot trajectory (or None)
            firing_order:  (B, K)   — when each slot fired (or None)
            total_loss:    scalar   — combined loss (if labels provided)
            loss_breakdown: dict    — per-component losses
        """
        B, L = input_ids.shape
        device = input_ids.device

        # ── 1. Embed tokens ─────────────────────────────────────────
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        x = self.embed_drop(x)

        # Compute padding mask: True for real tokens, False for PAD (id=0)
        pad_mask = input_ids != 0                     # (B, L)

        # ── 2. Mamba backbone ───────────────────────────────────────
        h_seq = self.backbone(x)                     # (B, L, d_model)

        # Last *real* token pooling (query-first: last real = last fact)
        # lengths(i) = number of non-PAD tokens in example i
        lengths = pad_mask.sum(dim=1).clamp(min=1)    # (B,)
        # Gather the hidden state at position lengths-1 for each example
        idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        idx = idx.expand(-1, 1, h_seq.size(-1))          # (B, 1, d_model)
        h_final = h_seq.gather(1, idx).squeeze(1)        # (B, d_model)

        # ── 3. Slot gating (if enabled) ─────────────────────────────
        slot_values = None
        slot_history = None
        firing_order = None

        if self.slot_gate is not None:
            # v10: set epoch for temperature annealing
            if hasattr(self.slot_gate, 'set_epoch'):
                self.slot_gate.set_epoch(current_epoch)
            slot_values, slot_history, firing_order = self.slot_gate(h_seq)

        # ── 4. Answer prediction ────────────────────────────────────
        if slot_values is not None:
            ans_input = torch.cat([h_final, slot_values], dim=-1)
        else:
            ans_input = h_final

        answer_logit = self.answer_head(ans_input).squeeze(-1)  # (B,)
        answer_prob = torch.sigmoid(answer_logit)               # (B,)

        # ── 5. Loss computation ─────────────────────────────────────
        total_loss = torch.tensor(0.0, device=device)
        loss_breakdown = {}

        if answer_labels is not None:
            l_task = self.task_loss_fn(answer_logit, answer_labels.float())
            total_loss = total_loss + l_task
            loss_breakdown["L_task"] = l_task.item()

        if self.logic_loss is not None and slot_values is not None:
            # Warmup: disable L_rule for first N epochs
            enable_rule = current_epoch >= self.cfg.slot_warmup_epochs
            l_logic, logic_bd = self.logic_loss(
                slot_preds=slot_values,
                slot_labels=slot_labels,
                answer_prob=answer_prob,
                rules=rules,
                per_example_rules=per_example_rules,
                enable_rule_loss=enable_rule,
            )
            total_loss = total_loss + l_logic
            loss_breakdown.update(logic_bd)
        elif self.logic_loss is not None and slot_values is None:
            # "loss_only" variant: construct pseudo-slots from answer_prob
            # Each pseudo-slot = answer_prob, so logic constraints operate
            # on predicted answers without explicit slot structure.
            K = self.cfg.n_slots
            pseudo_slots = answer_prob.unsqueeze(1).expand(-1, K)  # (B, K)
            enable_rule = current_epoch >= self.cfg.slot_warmup_epochs
            l_logic, logic_bd = self.logic_loss(
                slot_preds=pseudo_slots,
                slot_labels=None,       # no slot labels for loss_only
                answer_prob=answer_prob,
                rules=rules,
                per_example_rules=per_example_rules,
                enable_rule_loss=enable_rule,
            )
            total_loss = total_loss + l_logic
            loss_breakdown.update(logic_bd)

        loss_breakdown["L_total"] = total_loss.item()

        # v10: Add load-balancing loss for slot routing
        if (self.slot_gate is not None
                and getattr(self.slot_gate, 'routing', False)
                and answer_labels is not None):
            lambda_balance = getattr(self.cfg, 'lambda_balance', 0.01)
            l_balance = self.slot_gate.get_load_balance_loss()
            total_loss = total_loss + lambda_balance * l_balance
            loss_breakdown["L_balance"] = l_balance.item()
            loss_breakdown["L_total"] = total_loss.item()

        return {
            "answer_prob": answer_prob,
            "slot_values": slot_values,
            "slot_history": slot_history,
            "firing_order": firing_order,
            "total_loss": total_loss,
            "loss_breakdown": loss_breakdown,
        }

    @torch.no_grad()
    def explain(
        self,
        input_ids: torch.Tensor,
        rule_names: list[str] | None = None,
    ) -> list[dict]:
        """
        Inference with self-explanation.

        Returns list (per batch) of:
            {"answer": bool, "confidence": float, "trace": [...]}
        """
        self.eval()
        result = self(input_ids)

        explanations = []
        B = input_ids.shape[0]

        for b in range(B):
            entry = {
                "answer": result["answer_prob"][b].item() > 0.5,
                "confidence": round(result["answer_prob"][b].item(), 4),
                "trace": [],
            }
            if (
                self.slot_gate is not None
                and result["slot_values"] is not None
            ):
                traces = self.slot_gate.explain(
                    result["slot_values"][b:b+1],
                    result["firing_order"][b:b+1],
                    rule_names,
                )
                entry["trace"] = traces[0]
            explanations.append(entry)

        return explanations

    def slot_token_attribution(
        self,
        input_ids: torch.Tensor,   # (B, L)
        slot_idx: int = 0,         # which slot to attribute
        n_steps: int = 20,         # integration steps
    ) -> torch.Tensor:
        """
        Compute integrated-gradient attribution from slot `slot_idx`
        to each input token position.

        This creates a proper slot → token importance map, replacing
        the random masking heuristic. For each token position, the
        attribution score indicates how much that token contributes
        to the slot's activation.

        Args:
            input_ids: (B, L) input token IDs
            slot_idx: which slot to compute attribution for
            n_steps: number of interpolation steps for integration

        Returns:
            attributions: (B, L) importance scores per token position
        """
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device

        # Get embedding for baseline (zero embedding) and actual input
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x_embed = self.embedding(input_ids) + self.pos_embedding(positions)
        baseline = torch.zeros_like(x_embed)  # zero embedding baseline

        # Disable param gradients so IG only differentiates w.r.t. input
        param_grad_state = {}
        for name, p in self.named_parameters():
            param_grad_state[name] = p.requires_grad
            p.requires_grad_(False)

        # Accumulate gradients along interpolation path
        grads_sum = torch.zeros_like(x_embed)

        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            x_interp = baseline + alpha * (x_embed - baseline)
            x_interp = x_interp.detach().requires_grad_(True)

            # Forward pass using embedding directly (skip embedding layer)
            h_seq = self.backbone(self.embed_drop(x_interp))

            if self.slot_gate is not None:
                slot_values, _, _ = self.slot_gate(h_seq)
                target = slot_values[:, slot_idx].sum()
            else:
                # Fallback: attribute to answer probability
                pad_mask = input_ids != 0
                lengths = pad_mask.sum(dim=1).clamp(min=1)
                idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1)
                idx = idx.expand(-1, 1, h_seq.size(-1))
                h_final = h_seq.gather(1, idx).squeeze(1)
                answer_logit = self.answer_head(h_final).squeeze(-1)
                target = answer_logit.sum()

            # Use autograd.grad to avoid accumulating grads on model params
            grads = torch.autograd.grad(target, x_interp, retain_graph=False)[0]
            grads_sum += grads.detach()

        # Restore param gradient state
        for name, p in self.named_parameters():
            p.requires_grad_(param_grad_state[name])

        # Integrated gradients: (x - baseline) * avg_gradient
        avg_grad = grads_sum / n_steps
        ig = (x_embed.detach() - baseline) * avg_grad  # (B, L, d_model)

        # Reduce over embedding dim → (B, L) importance
        attributions = ig.norm(dim=-1)  # L2 norm per position

        # Normalise to [0, 1] per example
        max_attr = attributions.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        attributions = attributions / max_attr

        return attributions
