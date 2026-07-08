"""
Configuration dataclass for the Slot-Gated NeSy Mamba model.
"""

from dataclasses import dataclass, field
import math


@dataclass
class NeSyMambaConfig:
    """All hyperparameters for the NeSy-Mamba model."""

    # ── Mamba backbone ──────────────────────────────────────────────
    d_model: int = 128          # Hidden dimension
    d_state: int = 16           # SSM latent state dimension (N)
    d_conv: int = 4             # 1-D conv kernel width
    n_layers: int = 4           # Number of stacked Mamba blocks
    expand_factor: int = 2      # Inner dim = expand_factor * d_model
    dt_rank: int = 0            # Rank of Δ projection; 0 = auto

    # ── Token embedding ─────────────────────────────────────────────
    vocab_size: int = 5000      # Vocabulary size
    max_seq_len: int = 512      # Maximum sequence length

    # ── Symbolic slots ──────────────────────────────────────────────
    n_slots: int = 7            # K = N_rules + 2  (default: 5 rules + 2 buffer)
    slot_threshold: float = 0.5 # Threshold for "fired" in explanation trace

    # ── Logic loss weights ──────────────────────────────────────────
    lambda_slot: float = 1.0    # Slot supervision weight
    lambda_rule: float = 0.3    # Answer-consistency weight
    lambda_ortho: float = 0.1   # Orthogonality penalty weight
    lambda_entropy: float = 0.1  # Slot entropy regularisation (anti-collapse)

    # ── Slot initialisation ────────────────────────────────────
    slot_bias_init: float = -3.0       # Initial bias for slot gates (sigmoid(-3)≈0.05  → start OFF)
    slot_bias_stagger: float = 0.0     # v12: Spread biases by ±this amount (0=all same, 1.0=linspace(-1,+1))
    slot_alpha_stagger: float = 0.5    # v12: Spread alpha biases by ±this amount for symmetry breaking
    slot_recurrence_init: float = -0.1 # Diagonal recurrence init (negative = self-inhibiting)
    slot_ws_gain: float = 0.1          # Xavier gain for W_s projection

    # ── Slot gate mode ──────────────────────────────────────────────
    # "monotonic" — original max-based gate (once fired, stays)
    # "ema"       — exponential moving average gate (slots can rise AND fall)
    slot_gate_mode: str = "monotonic"
    slot_ema_alpha_init: float = 2.0   # logit init for EMA alpha (sigmoid(2)≈0.88 = slow update)
    slot_ema_dynamic_alpha: bool = True  # α_k(t) = σ(W_α·h_t + b_α) — input-dependent memory gate

    # ── Slot dropout ─────────────────────────────────────────────────
    slot_dropout: float = 0.0        # Dropout on slot activations (0 = disabled)

    # ── Slot routing (v10) ───────────────────────────────────────────
    slot_routing: bool = False          # Top-k hard routing: each token updates only k slots
    slot_routing_top_k: int = 2         # How many slots each token updates (MoE-style)
    lambda_balance: float = 0.01        # Load-balancing loss weight (prevents routing collapse)
    slot_ortho_init: bool = False       # Orthogonal W_s init (forces diverse slot projections)

    # ── Slot competition ────────────────────────────────────────────
    slot_competition: bool = False   # Softmax competition among slot candidates
    slot_temperature: float = 0.5    # Temperature for slot competition softmax
    coupled_slots: bool = False      # If True, use full K×K recurrence; else diagonal (per-slot scalar)

    # ── Regularisation ───────────────────────────────────────────
    dropout: float = 0.1            # Dropout between Mamba layers
    pos_weight: float = 1.0         # Class balancing: weight for positive (True) class

    # ── Training ────────────────────────────────────────────────────
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 32
    grad_clip: float = 1.0      # Max gradient norm
    slot_warmup_epochs: int = 10 # Epochs before enabling L_rule

    # ── Ablation variant ────────────────────────────────────────────
    # "base" | "slots_only" | "loss_only" | "full"
    variant: str = "full"

    # ── Additional training ─────────────────────────────────────────
    weight_decay: float = 0.01
    lr_schedule: str = "cosine"     # "cosine" | "constant"
    save_dir: str = "checkpoints"
    patience: int = 0               # Early stopping (0 = disabled)

    # ── Dataset ─────────────────────────────────────────────────────
    dataset: str = "synthetic"      # "synthetic" | "proofwriter" | "clutrr"
    data_dir: str = "data"

    # ── Transfer learning ──────────────────────────────────────────
    freeze_slots: bool = False

    def __post_init__(self):
        # Auto-compute dt_rank if not set
        if self.dt_rank == 0:
            self.dt_rank = math.ceil(self.d_model / 16)

    @property
    def d_inner(self) -> int:
        return self.expand_factor * self.d_model

    @property
    def use_slots(self) -> bool:
        return self.variant in ("slots_only", "full")

    @property
    def use_logic_loss(self) -> bool:
        return self.variant in ("loss_only", "full")

    def to_dict(self) -> dict:
        """Serialize config to dict."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NeSyMambaConfig":
        """Create config from dict (ignores unknown keys)."""
        import inspect
        valid = {k for k in inspect.signature(cls).parameters}
        return cls(**{k: v for k, v in d.items() if k in valid})
