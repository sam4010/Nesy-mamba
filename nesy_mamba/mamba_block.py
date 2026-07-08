"""
Pure-PyTorch Mamba Block — Selective State Space Model.

Sequential scan implementation (no custom CUDA kernels) for
maximum readability and portability. Based on Gu & Dao (2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import NeSyMambaConfig


# ── Helpers ─────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ── Selective SSM (sequential scan) ─────────────────────────────────

def selective_scan_sequential(
    x: torch.Tensor,       # (B, L, D)
    delta: torch.Tensor,   # (B, L, D)
    A: torch.Tensor,       # (D, N)
    B: torch.Tensor,       # (B, L, N)
    C: torch.Tensor,       # (B, L, N)
    D_param: torch.Tensor, # (D,)
) -> torch.Tensor:
    """
    Selective scan via sequential recurrence (no parallel scan).

    Discretisation:
        A_bar = exp(Δ · A)      — (B, L, D, N)
        B_bar = Δ · B           — (B, L, D, N)
    Recurrence:
        h[t] = A_bar[t] · h[t-1] + B_bar[t] · x[t]
        y[t] = C[t] · h[t]  +  D · x[t]
    """
    B_batch, L, D = x.shape
    N = A.shape[1]

    # Discretise: Δ is (B, L, D), A is (D, N)
    # delta_A: (B, L, D, N)
    delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    # delta_B: (B, L, D, N)
    delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)

    # Sequential scan
    h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
    ys = []

    for t in range(L):
        # h = A_bar * h + B_bar * x
        h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
        # y = C · h  (inner product over N)
        y_t = (C[:, t].unsqueeze(1) * h).sum(-1)  # (B, D)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)      # (B, L, D)
    y = y + x * D_param.unsqueeze(0).unsqueeze(0)
    return y


# ── Mamba Block ─────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Single Mamba block: projection → conv → selective SSM → gated output.
    """

    def __init__(self, cfg: NeSyMambaConfig):
        super().__init__()
        d = cfg.d_model
        d_in = cfg.d_inner
        N = cfg.d_state

        # Input projections: x → (z, x_proj) each of dim d_inner
        self.in_proj = nn.Linear(d, d_in * 2, bias=False)

        # 1-D depth-wise convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_in,
            out_channels=d_in,
            kernel_size=cfg.d_conv,
            padding=cfg.d_conv - 1,
            groups=d_in,
            bias=True,
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(d_in, cfg.dt_rank + N * 2, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank, d_in, bias=True)

        # ── Critical: Mamba-paper dt initialisation ─────────────────
        # Without this, delta ≈ 0.7 → A_bar ≈ 0.5 per step → signal
        # decays to 0 after ~10 tokens.  With this, delta ∈ [0.001, 0.1]
        # → A_bar ≈ 0.99 → 60 % signal retained over 50 steps.
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Small weight init for dt_proj (paper: dt_rank^{-0.5})
        dt_init_std = cfg.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Learnable SSM parameters
        # A is initialised as a structured log-space matrix
        A = torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).repeat(d_in, 1)
        self.A_log = nn.Parameter(torch.log(A))       # (d_inner, N)
        self.D = nn.Parameter(torch.ones(d_in))        # (d_inner,)

        # Output projection
        self.out_proj = nn.Linear(d_in, d, bias=False)

        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape
        N = self.cfg.d_state

        # 1) Project to 2 × d_inner
        xz = self.in_proj(x)                           # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)              # each (B, L, d_inner)

        # 2) Depth-wise conv (causal: trim future)
        x_conv = x_branch.transpose(1, 2)              # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]         # causal trim
        x_conv = x_conv.transpose(1, 2)                # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 3) Compute SSM params from convolved input
        x_ssm = self.x_proj(x_conv)                    # (B, L, dt_rank + 2N)
        dt, B_mat, C_mat = x_ssm.split(
            [self.cfg.dt_rank, N, N], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)

        A = -torch.exp(self.A_log)                      # (d_inner, N)

        # 4) Selective scan
        y = selective_scan_sequential(x_conv, delta, A, B_mat, C_mat, self.D)

        # 5) Gated output
        y = y * F.silu(z)
        return self.out_proj(y)


# ── Mamba Backbone (stacked blocks + norm) ──────────────────────────

class MambaBackbone(nn.Module):
    """
    Stack of N Mamba blocks with RMSNorm and residual connections.
    """

    def __init__(self, cfg: NeSyMambaConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": RMSNorm(cfg.d_model),
                "mamba": MambaBlock(cfg),
            }))
        self.final_norm = RMSNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)  — hidden states at every position.
        """
        for layer in self.layers:
            x = x + self.dropout(layer["mamba"](layer["norm"](x)))  # pre-norm + drop
        return self.final_norm(x)
