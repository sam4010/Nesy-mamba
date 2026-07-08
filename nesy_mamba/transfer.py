"""
Transfer Learning: ProofWriter → CLUTRR.

Protocol (from the design docs):
  1. Train full slot-gated model on ProofWriter.
  2. Freeze the slot gating module (slot semantics are locked).
  3. Fine-tune the encoder and answer head on CLUTRR.
  4. Evaluate on CLUTRR with held-out kinship combinations.

Usage:
  python -m nesy_mamba.transfer \
      --source_ckpt checkpoints/best_full.pt \
      --target_dataset clutrr \
      --data_dir data \
      --epochs 20
"""

import argparse
import os
import torch
import torch.nn as nn
from .config import NeSyMambaConfig
from .nesy_mamba import NeSyMamba
from .data_utils import get_dataloaders, CLUTRRDataset
from .metrics import compute_metrics
from .train import train_one_epoch, evaluate


def freeze_slot_gate(model: NeSyMamba):
    """Freeze all parameters in the slot gating module."""
    if model.slot_gate is not None:
        for param in model.slot_gate.parameters():
            param.requires_grad = False
        print("  Slot gate frozen (parameters locked).")
    else:
        print("  Warning: No slot gate to freeze (variant may not use slots).")


def unfreeze_slot_gate(model: NeSyMamba):
    """Unfreeze all parameters in the slot gating module."""
    if model.slot_gate is not None:
        for param in model.slot_gate.parameters():
            param.requires_grad = True
        print("  Slot gate unfrozen.")


def load_source_model(
    ckpt_path: str,
    device: torch.device,
    target_vocab_size: int | None = None,
    target_max_seq_len: int | None = None,
) -> NeSyMamba:
    """
    Load a pretrained NeSy Mamba model from checkpoint.

    If target_vocab_size or target_max_seq_len differ from source,
    the embedding layers are re-initialized (since vocabulary changes).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct config
    cfg_dict = ckpt["config"]
    if target_vocab_size is not None:
        cfg_dict["vocab_size"] = target_vocab_size
    if target_max_seq_len is not None:
        cfg_dict["max_seq_len"] = target_max_seq_len

    cfg = NeSyMambaConfig.from_dict(cfg_dict)
    model = NeSyMamba(cfg)

    # Load weights (partial — skip mismatched embedding layers)
    source_state = ckpt["model_state_dict"]
    model_state = model.state_dict()

    loaded, skipped = [], []
    for name, param in source_state.items():
        if name in model_state and param.shape == model_state[name].shape:
            model_state[name] = param
            loaded.append(name)
        else:
            skipped.append(name)

    model.load_state_dict(model_state)
    print(f"  Loaded {len(loaded)} params, skipped {len(skipped)} (shape mismatch)")
    if skipped:
        print(f"  Skipped: {skipped}")

    return model.to(device)


def transfer_train(
    model: NeSyMamba,
    train_loader,
    val_loader,
    answer_rules: list[tuple[int, ...]],
    epochs: int,
    lr: float,
    device: torch.device,
    grad_clip: float = 1.0,
    freeze_slots: bool = True,
):
    """Run the transfer learning fine-tuning loop."""

    if freeze_slots:
        freeze_slot_gate(model)

    # Only optimize unfrozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} parameters")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0

    header = (f"{'Ep':>3} | {'Loss':>8} | {'TrAcc':>6} | {'VlAcc':>6} | "
              f"{'VlLFid':>6} | {'VlSFid':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(epochs):
        train_m = train_one_epoch(
            model, train_loader, optimizer, answer_rules, epoch, device, grad_clip
        )
        val_m = evaluate(model, val_loader, answer_rules, device)
        scheduler.step()

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch = epoch
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_m,
                "config": vars(model.cfg),
            }, os.path.join("checkpoints", "best_transfer.pt"))

        print(
            f"{epoch+1:>3} | {train_m['loss']:>8.4f} | "
            f"{train_m['accuracy']:>6.3f} | "
            f"{val_m['accuracy']:>6.3f} | "
            f"{val_m['logic_fidelity']:>6.3f} | "
            f"{val_m.get('slot_fidelity', 0):>6.3f}"
        )

    # Load best
    ckpt_path = os.path.join("checkpoints", "best_transfer.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    final_val = evaluate(model, val_loader, answer_rules, device)

    print(f"\n  Transfer Results (best epoch {best_epoch + 1}):")
    print(f"    Accuracy:       {final_val['accuracy']:.4f}")
    print(f"    Logic Fidelity: {final_val['logic_fidelity']:.4f}")
    print(f"    Slot Fidelity:  {final_val.get('slot_fidelity', 0):.4f}")

    return final_val


def parse_args():
    p = argparse.ArgumentParser(description="Transfer NeSy Mamba (ProofWriter → CLUTRR)")
    p.add_argument("--source_ckpt", type=str, required=True,
                   help="Path to pretrained model checkpoint")
    p.add_argument("--target_dataset", type=str, default="clutrr",
                   choices=["clutrr", "synthetic"])
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--freeze_slots", action="store_true", default=True,
                   help="Freeze slot gate during fine-tuning (recommended)")
    p.add_argument("--no_freeze_slots", dest="freeze_slots", action="store_false",
                   help="Fine-tune slot gate too (full fine-tuning)")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Source: {args.source_ckpt}")
    print(f"Target: {args.target_dataset}")
    print(f"Freeze slots: {args.freeze_slots}")

    # Load source model
    print(f"\n{'='*60}")
    print("  Loading pretrained model...")
    print(f"{'='*60}")
    model = load_source_model(args.source_ckpt, device)

    # Load target data
    print(f"\n  Loading {args.target_dataset} data...")

    if args.target_dataset == "clutrr":
        answer_rules = CLUTRRDataset.get_answer_rules()
        train_loader, val_loader = get_dataloaders(
            dataset_name="clutrr",
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            max_seq_len=model.cfg.max_seq_len,
            n_slots=model.cfg.n_slots,
        )
    else:
        # For testing: transfer to a different synthetic split
        from .data_utils import SyntheticRulesDataset
        answer_rules = SyntheticRulesDataset.get_answer_rules()
        train_loader, val_loader = get_dataloaders(
            dataset_name="synthetic",
            batch_size=args.batch_size,
            n_train=1000,
            n_val=300,
            seq_len=model.cfg.max_seq_len,
            vocab_size=model.cfg.vocab_size,
            n_slots=model.cfg.n_slots,
        )

    # Transfer fine-tuning
    print(f"\n{'='*60}")
    print("  Transfer Fine-Tuning")
    print(f"{'='*60}")

    transfer_train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        answer_rules=answer_rules,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        freeze_slots=args.freeze_slots,
    )

    print(f"\n{'='*60}")
    print("  TRANSFER COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
