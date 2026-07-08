"""
Symbolic Slot Gating Module.

Each slot is a sigmoid-gated truth-flag that persists through time.
Supports two gate modes:

  - **monotonic** (original): once activated, stays activated via max gate.
    s_k(t) = max(s_k(t-1), candidate_k(t))
    Good for proof accumulation but prone to all-slots-same-value collapse.

  - **ema** (v8): exponential moving average with per-slot alpha gate.
    Static:  s_k(t) = α_k * s_k(t-1) + (1 - α_k) * candidate_k(t)
    Dynamic: α_k(t) = σ(W_α · h_t + b_α_k)  — input-dependent memory gate
    Allows slots to both rise AND fall, enabling input-dependent differentiation.
    Dynamic alpha makes slots behave as learnable symbolic filters that
    decide per-token how persistent each slot should be.

Slot recurrence modes:
  - Diagonal (default): each slot's recurrence is a per-slot scalar
    u_i * s_i(t-1), preserving per-slot identity for interpretability.
  - Coupled (cfg.coupled_slots=True): full K×K matrix U_s allows
    cross-slot interaction. Useful as ablation baseline.

Optional softmax competition: before the gate update, apply a
temperature-scaled softmax over candidate gate values to encourage
specialisation (only one slot updates per step).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import NeSyMambaConfig


class SlotGate(nn.Module):
    """
    K sigmoid-gated symbolic truth-flag slots.

    Update rule (per slot i, per timestep t):
        Diagonal mode (default):
            candidate_i(t) = σ( W_s · h_t + u_i · s_i(t−1) + b_s_i )
        Coupled mode:
            candidate_i(t) = σ( W_s · h_t + (U_s · s(t−1))_i + b_s_i )

        If competition=True:
            weight_i(t) = softmax(candidate / τ)_i
            gated_i(t) = weight_i(t) * candidate_i(t)
        Else:
            gated_i(t) = candidate_i(t)

        Gate mode:
            monotonic: s_i(t) = max(s_prev_i, gated_i(t))
            ema:       s_i(t) = α_i(t) · s_prev_i + (1 − α_i(t)) · gated_i(t)
                       where α_i(t) = σ(W_α · h_t + b_α_i) if dynamic, else σ(logit_i)

    Properties:
        - Each slot value ∈ [0, 1], interpretable as soft truth probability
        - Monotonic mode: once a fact is deduced, it persists (no forget gate)
        - EMA mode: facts can accumulate or retract (learnable memory per slot)
        - Tracks firing order: timestep when each slot first crosses threshold
        - Diagonal recurrence preserves per-slot identity (interpretability)
        - Optional softmax competition among slot candidates

    Args:
        cfg: NeSyMambaConfig with d_model, n_slots, slot_competition,
             slot_temperature, coupled_slots, slot_gate_mode.
    """

    def __init__(self, cfg: NeSyMambaConfig):
        super().__init__()
        K = cfg.n_slots
        d = cfg.d_model

        self.gate_mode = getattr(cfg, "slot_gate_mode", "monotonic")
        assert self.gate_mode in ("monotonic", "ema", "gru"), \
            f"Unknown slot_gate_mode: {self.gate_mode}"

        # Learned projection from hidden state
        ws_gain = getattr(cfg, "slot_ws_gain", 0.1)
        self.W_s = nn.Linear(d, K, bias=False)
        # v10: Orthogonal init forces each slot to read different features
        if getattr(cfg, 'slot_ortho_init', False):
            nn.init.orthogonal_(self.W_s.weight)
            self.W_s.weight.data.mul_(ws_gain)
        else:
            nn.init.xavier_uniform_(self.W_s.weight, gain=ws_gain)

        # Slot recurrence: diagonal (default) or full K×K (coupled)
        self.coupled = getattr(cfg, "coupled_slots", False)
        if self.coupled:
            # Full K×K recurrence — cross-slot interaction
            self.U_s = nn.Linear(K, K, bias=False)
            nn.init.xavier_uniform_(self.U_s.weight, gain=0.1)
        else:
            # Diagonal: per-slot scalar u_i ∈ ℝ^K
            # Each slot's recurrence is independent: u_i * s_i(t-1)
            recurrence_init = getattr(cfg, 'slot_recurrence_init', -0.1)
            self.u_diag = nn.Parameter(torch.full((K,), recurrence_init))

        bias_init = getattr(cfg, 'slot_bias_init', -3.0)
        # v12: Staggered bias init — each slot starts at a DIFFERENT sigmoid
        # operating point to break gradient symmetry and prevent collapse.
        # E.g. with bias_init=0.0 and stagger=1.0:
        #   slot biases = [-1.0, -0.67, -0.33, 0.0, +0.33, +0.67, +1.0]
        #   sigmoid     = [0.27,  0.34,  0.42, 0.5,  0.58,  0.66,  0.73]
        bias_stagger = getattr(cfg, 'slot_bias_stagger', 0.0)
        if bias_stagger > 0:
            spread = torch.linspace(-bias_stagger, bias_stagger, K)
            self.b_s = nn.Parameter(torch.full((K,), bias_init) + spread)
        else:
            self.b_s = nn.Parameter(torch.full((K,), bias_init))

        # EMA mode: alpha gate (controls memory vs update)
        self.dynamic_alpha = getattr(cfg, "slot_ema_dynamic_alpha", True)
        if self.gate_mode == "ema":
            alpha_init = getattr(cfg, "slot_ema_alpha_init", 2.0)
            if self.dynamic_alpha:
                # Input-dependent: α_k(t) = σ(W_α · h_t + b_α_k)
                # This is like a GRU update gate — each token dynamically
                # controls how much each slot retains vs updates.
                self.W_alpha = nn.Linear(d, K)  # includes bias
                # Bias init: sigmoid(alpha_init) ≈ 0.88 → high memory by default
                nn.init.xavier_uniform_(self.W_alpha.weight, gain=0.1)
                # Stagger biases for symmetry breaking
                alpha_stagger = getattr(cfg, 'slot_alpha_stagger', 0.5)
                spread = torch.linspace(-alpha_stagger, alpha_stagger, K)
                self.W_alpha.bias.data.copy_(
                    torch.full((K,), alpha_init) + spread
                )
            else:
                # Static: per-slot learnable α (for ablation)
                spread = torch.linspace(-0.5, 0.5, K)  # symmetry-breaking
                self.alpha_logit = nn.Parameter(
                    torch.full((K,), alpha_init) + spread
                )

        # GRU mode: proper GRU update with reset gate (Slot Attention-style)
        # z = σ(W_z·h + U_z·s + b_z)    — update gate (like EMA alpha)
        # r = σ(W_r·h + U_r·s + b_r)    — reset gate (NEW: can erase memory)
        # candidate = tanh(W_c·h + U_c·(r⊙s) + b_c)  — gated candidate
        # s_new = z⊙s + (1-z)⊙candidate
        if self.gate_mode == "gru":
            alpha_init = getattr(cfg, "slot_ema_alpha_init", 2.0)
            alpha_stagger = getattr(cfg, 'slot_alpha_stagger', 0.5)
            spread = torch.linspace(-alpha_stagger, alpha_stagger, K)

            # Update gate: z_k(t) = σ(W_z·h_t + U_z·s(t-1) + b_z_k)
            self.W_z = nn.Linear(d, K, bias=False)
            self.U_z = nn.Linear(K, K, bias=False)
            self.b_z = nn.Parameter(torch.full((K,), alpha_init) + spread)
            nn.init.xavier_uniform_(self.W_z.weight, gain=0.1)
            nn.init.zeros_(self.U_z.weight)  # start with no s_prev influence

            # Reset gate: r_k(t) = σ(W_r·h_t + U_r·s(t-1) + b_r_k)
            # Init bias positive → reset starts high (allow full memory readout)
            self.W_r = nn.Linear(d, K, bias=False)
            self.U_r = nn.Linear(K, K, bias=False)
            self.b_r = nn.Parameter(torch.full((K,), 2.0) + spread)  # σ(2)=0.88 → mostly open
            nn.init.xavier_uniform_(self.W_r.weight, gain=0.1)
            nn.init.zeros_(self.U_r.weight)

            # Candidate: tanh(W_c·h_t + U_c·(r⊙s) + b_c_k)
            self.W_c = nn.Linear(d, K, bias=False)
            self.U_c = nn.Linear(K, K, bias=False)
            self.b_c = nn.Parameter(torch.zeros(K) + spread * 0.5)  # diverse starting points
            nn.init.xavier_uniform_(self.W_c.weight, gain=ws_gain)
            nn.init.zeros_(self.U_c.weight)

            # Orthogonal init for W_c if requested (like W_s for EMA)
            if getattr(cfg, 'slot_ortho_init', False):
                nn.init.orthogonal_(self.W_c.weight)
                self.W_c.weight.data.mul_(ws_gain)

        self.threshold = cfg.slot_threshold
        self.n_slots = K
        self.competition = getattr(cfg, "slot_competition", False)
        self.temperature = getattr(cfg, "slot_temperature", 0.5)
        self.slot_dropout = getattr(cfg, "slot_dropout", 0.0)

        # v10: Top-K hard slot routing (MoE-style)
        self.routing = getattr(cfg, 'slot_routing', False)
        self.top_k = getattr(cfg, 'slot_routing_top_k', 2)
        if self.routing:
            # Separate router head: W_r ∈ ℝ^{d × K}
            self.W_router = nn.Linear(d, K, bias=True)
            nn.init.xavier_uniform_(self.W_router.weight, gain=0.5)
            nn.init.zeros_(self.W_router.bias)

        # Track router load balance across forward pass (for aux loss)
        self._router_load: torch.Tensor | None = None  # (K,) fraction of tokens routed to each slot
        self._router_probs: torch.Tensor | None = None  # (K,) mean router probability per slot

    def get_load_balance_loss(self) -> torch.Tensor:
        """
        Compute Switch Transformer-style load-balancing loss.

        L_balance = K * sum_k(f_k * p_k)
        where f_k = fraction of tokens routed to slot k
              p_k = mean router probability for slot k

        Returns 0 if routing is disabled or no forward pass has been done.
        """
        if self._router_load is None or self._router_probs is None:
            return torch.tensor(0.0)
        K = self._router_load.shape[0]
        # f_k * p_k summed over slots, then scaled by K
        return K * (self._router_load * self._router_probs).sum()

    def forward(
        self,
        h_seq: torch.Tensor,   # (B, L, d_model)  — Mamba hidden states
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the full sequence and update slots at every timestep.

        Returns:
            slot_values:  (B, K)     — final slot activation values
            slot_history: (B, L, K)  — slot values at every timestep
            firing_order: (B, K)     — timestep when each slot first fired
                                       (-1 if never fired)
        """
        B, L, _ = h_seq.shape
        K = self.n_slots
        device = h_seq.device
        dtype = h_seq.dtype

        # Initialise slots to 0 (all facts unknown)
        s = torch.zeros(B, K, device=device, dtype=dtype)

        # Pre-compute static EMA alpha if applicable
        if self.gate_mode == "ema" and not self.dynamic_alpha:
            alpha = torch.sigmoid(self.alpha_logit)  # (K,) ∈ (0, 1)

        # Track history and firing order
        history = []
        firing_order = torch.full(
            (B, K), -1, device=device, dtype=torch.long
        )

        # v10: Track routing statistics for load-balance loss
        if self.routing:
            route_counts = torch.zeros(K, device=device, dtype=dtype)  # tokens per slot
            route_probs_sum = torch.zeros(K, device=device, dtype=dtype)  # sum of router probs
            total_tokens = 0

        for t in range(L):
            h_t = h_seq[:, t, :]                        # (B, d_model)

            # Recurrence term: diagonal or coupled
            if self.coupled:
                recurrence = self.U_s(s)                 # (B, K)
            else:
                recurrence = self.u_diag * s             # (B, K) element-wise

            # Raw logits and candidate slot values
            logits = self.W_s(h_t) + recurrence + self.b_s  # (B, K)
            candidate = torch.sigmoid(logits)                # (B, K)

            # v11: Compute routing mask (determines WHICH slots update)
            if self.routing:
                router_logits = self.W_router(h_t)               # (B, K)
                router_probs = F.softmax(router_logits, dim=-1)  # (B, K)

                # Top-k selection per example
                _, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (B, top_k)

                # Build hard mask: 1 for selected slots, 0 for others
                mask = torch.zeros_like(router_probs)            # (B, K)
                mask.scatter_(1, topk_idx, 1.0)

                # Straight-through: hard mask in forward, soft probs in backward
                routing_mask = mask - router_probs.detach() + router_probs  # (B, K)

                # Accumulate load-balance stats
                route_counts += mask.sum(dim=0)                  # (K,)
                route_probs_sum += router_probs.sum(dim=0)       # (K,)
                total_tokens += B

            # Candidate gating (routing does NOT multiply candidate — v11 fix)
            if self.competition and not self.routing:
                weights = torch.softmax(candidate / self.temperature, dim=-1)
                gated = weights * candidate                      # (B, K)
            else:
                gated = candidate

            # Slot dropout: prevent over-reliance on single slots
            if self.slot_dropout > 0 and self.training:
                gated = F.dropout(gated, p=self.slot_dropout, training=True)

            # Gate update → compute what new slot values WOULD be
            if self.gate_mode == "monotonic":
                new_s = torch.max(s, gated)
            elif self.gate_mode == "gru":
                # GRU update: reset gate allows erasing old state
                z = torch.sigmoid(self.W_z(h_t) + self.U_z(s) + self.b_z)  # update gate (B,K)
                r = torch.sigmoid(self.W_r(h_t) + self.U_r(s) + self.b_r)  # reset gate (B,K)
                s_reset = r * s                                             # gated memory
                c = torch.tanh(self.W_c(h_t) + self.U_c(s_reset) + self.b_c)  # candidate
                c = (c + 1.0) / 2.0  # scale tanh [-1,1] → [0,1] for slot values
                new_s = z * s + (1.0 - z) * c
            else:
                # EMA: slots can rise AND fall based on input
                if self.dynamic_alpha:
                    alpha = torch.sigmoid(self.W_alpha(h_t))     # (B, K)
                new_s = alpha * s + (1.0 - alpha) * gated

            # v11 route-then-hold: only update selected slots, hold others
            if self.routing:
                s = routing_mask * new_s + (1.0 - routing_mask) * s
            else:
                s = new_s

            history.append(s)

            # Record first firing time
            newly_fired = (s > self.threshold) & (firing_order < 0)
            firing_order[newly_fired] = t

        slot_history = torch.stack(history, dim=1)       # (B, L, K)
        slot_values = s                                  # (B, K) — final

        # v10: Store router statistics for load-balance loss computation
        if self.routing and total_tokens > 0:
            self._router_load = route_counts / (total_tokens + 1e-8)   # (K,) fraction
            self._router_probs = route_probs_sum / (total_tokens + 1e-8)  # (K,) mean prob
        else:
            self._router_load = None
            self._router_probs = None

        return slot_values, slot_history, firing_order

    def explain(
        self,
        slot_values: torch.Tensor,    # (B, K)
        firing_order: torch.Tensor,   # (B, K)
        rule_names: list[str] | None = None,
    ) -> list[list[dict]]:
        """
        Generate human-readable explanation traces.

        Returns:
            List (per batch) of lists of dicts:
            [{"slot": i, "rule": name, "step": t, "value": v}, ...]
            sorted by firing order.
        """
        B, K = slot_values.shape
        if rule_names is None:
            rule_names = [f"Rule_{i}" for i in range(K)]

        explanations = []
        for b in range(B):
            trace = []
            for k in range(K):
                if slot_values[b, k].item() > self.threshold:
                    trace.append({
                        "slot": k,
                        "rule": rule_names[k],
                        "step": firing_order[b, k].item(),
                        "value": round(slot_values[b, k].item(), 4),
                    })
            # Sort by firing order (step)
            trace.sort(key=lambda x: x["step"])
            explanations.append(trace)

        return explanations
