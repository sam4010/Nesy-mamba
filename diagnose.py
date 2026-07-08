"""
Diagnostic: Why is NeSy-Mamba stuck at random chance on ProofWriter?

HYPOTHESIS: The dt_proj bias in MambaBlock uses default PyTorch init,
giving delta ~ 0.7.  With A = -1 (slowest decay channel):
    A_bar = exp(delta * A) = exp(-0.7) ~ 0.50
    After 50 steps:  0.50^50 ~ 10^{-15}   <-- ALL query information GONE

The original Mamba paper initialises dt so that delta in [0.001, 0.1]:
    A_bar = exp(-0.01) ~ 0.99
    After 50 steps:  0.99^50 ~ 0.60       <-- 60% signal retained

This script measures the actual delta values and hidden-state decay
in our current model.
"""

import math
import torch
import torch.nn.functional as F
from nesy_mamba.config import NeSyMambaConfig
from nesy_mamba.mamba_block import MambaBlock, MambaBackbone
from nesy_mamba.nesy_mamba import NeSyMamba

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def diagnose_delta_init():
    """Show what delta values our current model produces."""
    print("=" * 65)
    print("  DIAGNOSIS 1: What delta values does our Mamba produce?")
    print("=" * 65)

    cfg = NeSyMambaConfig(d_model=128, n_layers=1, max_seq_len=64, vocab_size=50)
    block = MambaBlock(cfg).to(device)

    # Random input (B=4, L=64, d_model=128)
    x = torch.randn(4, 64, 128, device=device)

    # Trace through the block to get delta values
    with torch.no_grad():
        d_in = cfg.d_inner  # 256
        N = cfg.d_state     # 16

        xz = block.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2)
        x_conv = block.conv1d(x_conv)[:, :, :64]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_ssm = block.x_proj(x_conv)
        dt_raw, B_mat, C_mat = x_ssm.split([cfg.dt_rank, N, N], dim=-1)
        delta = F.softplus(block.dt_proj(dt_raw))    # (B, L, d_inner)

    print(f"\n  Config: d_model={cfg.d_model}, d_inner={cfg.d_inner}, "
          f"d_state={cfg.d_state}, dt_rank={cfg.dt_rank}")

    print(f"\n  dt_proj.bias stats:")
    bias = block.dt_proj.bias.data
    print(f"    mean={bias.mean():.4f}, std={bias.std():.4f}, "
          f"min={bias.min():.4f}, max={bias.max():.4f}")

    print(f"\n  Actual delta (after softplus) stats:")
    print(f"    mean={delta.mean():.4f}, std={delta.std():.4f}")
    print(f"    min={delta.min():.4f}, max={delta.max():.4f}")
    print(f"    median={delta.median():.4f}")

    # Compute A_bar for the slowest-decaying channel (A = -1)
    A = -torch.exp(block.A_log)
    a_min = A.max().item()            # closest to 0 = slowest decay
    a_max = A.min().item()            # most negative = fastest decay
    print(f"\n  A range: [{a_max:.2f}, {a_min:.2f}]")

    delta_mean = delta.mean().item()
    a_bar_slow = math.exp(delta_mean * a_min)
    a_bar_fast = math.exp(delta_mean * a_max)

    print(f"\n  With mean delta={delta_mean:.4f} and A=-1 (slowest decay):")
    print(f"    A_bar = exp({delta_mean:.4f} * -1) = {a_bar_slow:.6f}")
    print(f"    After 10 steps: {a_bar_slow**10:.6e}")
    print(f"    After 30 steps: {a_bar_slow**30:.6e}")
    print(f"    After 50 steps: {a_bar_slow**50:.6e}")

    print(f"\n  With mean delta={delta_mean:.4f} and A=-16 (fastest decay):")
    print(f"    A_bar = exp({delta_mean:.4f} * -16) = {a_bar_fast:.6e}")
    print(f"    After 5 steps: {a_bar_fast**5:.6e}")

    # What the Mamba paper uses
    print(f"\n  --- COMPARISON: Mamba paper init ---")
    dt_paper = 0.01
    a_bar_paper = math.exp(dt_paper * (-1))
    print(f"    delta=0.01, A=-1: A_bar = {a_bar_paper:.6f}")
    print(f"    After 50 steps: {a_bar_paper**50:.6f}")
    print(f"    Signal retained: {a_bar_paper**50*100:.1f}%")

    return delta_mean


def diagnose_hidden_state_decay():
    """Show how backbone hidden states decay through the sequence."""
    print("\n" + "=" * 65)
    print("  DIAGNOSIS 2: Does query information survive to fact positions?")
    print("=" * 65)

    cfg = NeSyMambaConfig(
        d_model=128, n_layers=4, max_seq_len=64, vocab_size=50
    )
    model = NeSyMamba(cfg).to(device)
    model.eval()

    # Two sequences: differ ONLY in query (pos 0-1), identical facts (pos 3+)
    base = torch.randint(1, 50, (1, 64), device=device)

    seq_a = base.clone()
    seq_a[0, 0] = 10;  seq_a[0, 1] = 20  # query A

    seq_b = base.clone()
    seq_b[0, 0] = 30;  seq_b[0, 1] = 40  # query B

    seq_a[0, 2] = 5;   seq_b[0, 2] = 5   # same [SEP]

    with torch.no_grad():
        positions = torch.arange(64, device=device).unsqueeze(0)
        emb_a = model.embedding(seq_a) + model.pos_embedding(positions)
        emb_b = model.embedding(seq_b) + model.pos_embedding(positions)
        h_a = model.backbone(emb_a)
        h_b = model.backbone(emb_b)

    diff = (h_a - h_b).norm(dim=-1).squeeze()
    h_norm = ((h_a.norm(dim=-1) + h_b.norm(dim=-1)) / 2).squeeze()
    rel_diff = diff / (h_norm + 1e-8)

    print(f"\n  Two sequences differ ONLY at positions 0-1 (query tokens).")
    print(f"  Positions 3-63 are identical fact tokens.")
    print(f"  If query info is retained, hidden states should DIFFER.")
    print(f"  If decayed, they will be IDENTICAL.\n")

    print(f"  {'Pos':>4} | {'||h_a-h_b||':>12} | {'||h|| avg':>10} | {'RelDiff':>8}")
    print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}")

    for pos in [0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 63]:
        if pos < 64:
            tag = " <-- query" if pos < 2 else (" <-- SEP" if pos == 2 else "")
            print(f"  {pos:>4} | {diff[pos]:>12.6f} | {h_norm[pos]:>10.4f} | "
                  f"{rel_diff[pos]:>8.6f}{tag}")

    if rel_diff[30] < 0.01:
        print(f"\n  ** VERDICT: Query info DECAYED TO ZERO by position 30.")
        print(f"  ** The model CANNOT distinguish queries over same facts.")
        print(f"  ** This explains accuracy = majority class = ~54.5%")
    elif rel_diff[30] < 0.1:
        print(f"\n  ** VERDICT: Query info is VERY WEAK by position 30.")
    else:
        print(f"\n  ** Query info is partially retained at position 30.")


def diagnose_overfit():
    """Can the base model overfit 20 random examples?"""
    print("\n" + "=" * 65)
    print("  DIAGNOSIS 3: Can the model overfit 20 random examples?")
    print("=" * 65)

    cfg = NeSyMambaConfig(
        d_model=128, n_layers=4, max_seq_len=32, vocab_size=50,
        n_slots=7, variant="base"
    )
    model = NeSyMamba(cfg).to(device)

    X = torch.randint(1, 50, (20, 32), device=device)
    Y = torch.randint(0, 2, (20,), device=device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n  20 random examples, d_model=128, 4 layers, base variant")
    print(f"  If it can't memorise random labels -> architecture is broken.\n")

    for ep in range(100):
        model.train()
        result = model(X, answer_labels=Y)
        loss = result["total_loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            preds = (result["answer_prob"] > 0.5).float()
            acc = (preds == Y).float().mean().item()

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1:>3}: loss={loss.item():.4f}, acc={acc:.3f}")

        if acc >= 1.0:
            print(f"  -> Perfect overfit at epoch {ep+1}!")
            return True

    final_acc = acc
    print(f"\n  Final accuracy after 100 epochs: {final_acc:.3f}")
    if final_acc < 0.8:
        print(f"  ** VERDICT: Can't even memorise 20 examples -> architecture issue")
    return False


def show_fix():
    """Show what proper dt initialisation looks like."""
    print("\n" + "=" * 65)
    print("  FIX: Proper dt_proj initialisation (from Mamba paper)")
    print("=" * 65)

    cfg = NeSyMambaConfig(d_model=128, n_layers=1, max_seq_len=64, vocab_size=50)
    block = MambaBlock(cfg)
    d_in = cfg.d_inner

    bias_before = block.dt_proj.bias.data.clone()

    # Paper init
    dt_min, dt_max = 0.001, 0.1
    dt = torch.exp(
        torch.rand(d_in) * (math.log(dt_max) - math.log(dt_min))
        + math.log(dt_min)
    ).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))

    print(f"\n  CURRENT dt_proj.bias:")
    print(f"    mean={bias_before.mean():.4f}, range=[{bias_before.min():.4f}, {bias_before.max():.4f}]")
    print(f"    softplus -> delta ~ {F.softplus(bias_before).mean():.4f}")

    print(f"\n  PAPER dt_proj.bias:")
    print(f"    mean={inv_dt.mean():.4f}, range=[{inv_dt.min():.4f}, {inv_dt.max():.4f}]")
    print(f"    softplus -> delta ~ {F.softplus(inv_dt).mean():.4f}")

    delta_paper = F.softplus(inv_dt).mean().item()
    a_bar = math.exp(delta_paper * (-1))
    print(f"\n  With paper init, slowest channel (A=-1):")
    print(f"    A_bar per step = {a_bar:.6f}")
    print(f"    After 50 steps = {a_bar**50:.6f}  ({a_bar**50*100:.1f}% retained)")


if __name__ == "__main__":
    diagnose_delta_init()
    diagnose_hidden_state_decay()
    diagnose_overfit()
    show_fix()
