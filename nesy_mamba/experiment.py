"""
Ablation Experiment Runner for NeSy Mamba.

Runs all 4 ablation variants and produces a comparison table:
  base       — vanilla Mamba (no slots, no logic loss)
  slots_only — Mamba + slots (no logic loss)
  loss_only  — Mamba + logic loss (no explicit slots)
  full       — Mamba + slots + logic losses

Usage:
  python -m nesy_mamba.experiment [--epochs 30] [--d_model 64] [--n_layers 2]
"""

import argparse
import os
import json
import torch
from .config import NeSyMambaConfig
from .nesy_mamba import NeSyMamba
from .data_utils import get_dataloaders, SyntheticRulesDataset
from .metrics import compute_metrics
from .train import train_one_epoch, evaluate


VARIANTS = ["base", "slots_only", "loss_only", "full"]


def run_single_variant(
    variant: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 32,
    d_model: int = 64,
    n_layers: int = 2,
    n_slots: int = 7,
    slot_warmup: int = 10,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train one variant and return final metrics + history."""

    cfg = NeSyMambaConfig(
        variant=variant,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_layers=n_layers,
        d_model=d_model,
        n_slots=n_slots,
        slot_warmup_epochs=slot_warmup,
        vocab_size=50,
        max_seq_len=32,
    )

    train_loader, val_loader = get_dataloaders(
        dataset_name="synthetic",
        batch_size=cfg.batch_size,
        n_train=2000,
        n_val=500,
        seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        n_slots=cfg.n_slots,
    )

    answer_rules = SyntheticRulesDataset.get_answer_rules()

    model = NeSyMamba(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        train_m = train_one_epoch(
            model, train_loader, optimizer, answer_rules, epoch, device, cfg.grad_clip
        )
        val_m = evaluate(model, val_loader, answer_rules, device)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch = epoch
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_m,
                "config": vars(cfg),
            }, os.path.join("checkpoints", f"best_{variant}.pt"))

    # Load best model
    ckpt_path = os.path.join("checkpoints", f"best_{variant}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final_val = evaluate(model, val_loader, answer_rules, device)

    return {
        "variant": variant,
        "n_params": n_params,
        "best_epoch": best_epoch + 1,
        "final_val": final_val,
        "history": history,
    }


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table of all variants."""
    print(f"\n{'='*72}")
    print("  ABLATION COMPARISON")
    print(f"{'='*72}")
    print(
        f"  {'Variant':<12} | {'Params':>8} | {'Acc':>6} | "
        f"{'LogFid':>6} | {'SlotFid':>7} | {'BestEp':>6}"
    )
    print(f"  {'-'*62}")

    for r in results:
        v = r["final_val"]
        print(
            f"  {r['variant']:<12} | {r['n_params']:>8,} | "
            f"{v['accuracy']:>6.3f} | "
            f"{v['logic_fidelity']:>6.3f} | "
            f"{v.get('slot_fidelity', 0):>7.3f} | "
            f"{r['best_epoch']:>6}"
        )

    print(f"{'='*72}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Run NeSy Mamba ablation study")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=7)
    p.add_argument("--slot_warmup", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--variants", type=str, nargs="+", default=VARIANTS,
                   choices=VARIANTS, help="Which variants to run")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Variants to run: {args.variants}")
    print(f"Epochs: {args.epochs}, d_model: {args.d_model}, n_layers: {args.n_layers}")

    all_results = []

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"  Running variant: {variant.upper()}")
        print(f"{'='*60}")

        result = run_single_variant(
            variant=variant,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_slots=args.n_slots,
            slot_warmup=args.slot_warmup,
            device=device,
        )
        all_results.append(result)

        v = result["final_val"]
        print(f"  {variant}: Acc={v['accuracy']:.3f}, "
              f"LogFid={v['logic_fidelity']:.3f}, "
              f"SlotFid={v.get('slot_fidelity', 0):.3f}")

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "ablation_results.json")
    serializable = []
    for r in all_results:
        serializable.append({
            "variant": r["variant"],
            "n_params": r["n_params"],
            "best_epoch": r["best_epoch"],
            "final_val": r["final_val"],
        })
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Results saved to {save_path}")

    return all_results


if __name__ == "__main__":
    main()
