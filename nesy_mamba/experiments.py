"""
Experiment Runner for NeSy Mamba.

Provides automated experiment suites for IEEE-level evaluation:
  1. Ablation Study  — base / slots_only / loss_only / full
  2. Compositional Generalisation — train ≤ depth D, test depth > D
  3. Fact-Order Robustness — shuffle_facts vs. original order
  4. Per-depth Accuracy Breakdown — depth 0, 1, 2, 3 separately
  5. Firing Order Correlation — slot firing time vs proof depth

Usage:
    python -m nesy_mamba.experiments --suite ablation --data_dir /path/to/data
    python -m nesy_mamba.experiments --suite all --data_dir /path/to/data
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn

from .config import NeSyMambaConfig
from .nesy_mamba import NeSyMamba
from .data_utils import (
    get_dataloaders,
    SymbolicProofWriterDataset,
    collate_fn,
)
from .metrics import compute_metrics
from .visualize import (
    plot_ablation_comparison,
    plot_firing_order_vs_depth,
    plot_per_depth_accuracy,
    compute_firing_depth_correlation,
)


# ── Shared Training Utilities ───────────────────────────────────────

def _build_model_and_train(
    cfg: NeSyMambaConfig,
    train_loader,
    val_loader,
    answer_rules,
    device: torch.device,
    verbose: bool = True,
) -> tuple[NeSyMamba, dict]:
    """
    Build model, train, return (model, history).
    Uses flat LR (constant, no scheduler) — the recipe that works best.
    """
    model = NeSyMamba(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=0.005,
    )

    history = {"train": [], "val": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        # ── Train ───────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            answer_label = batch["answer_label"].to(device)
            slot_labels = batch["slot_labels"].to(device)

            optimizer.zero_grad()
            result = model(
                input_ids=input_ids,
                answer_labels=answer_label,
                slot_labels=slot_labels if cfg.use_slots else None,
                rules=answer_rules,
                current_epoch=epoch,
            )
            result["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            with torch.no_grad():
                m = compute_metrics(
                    answer_prob=result["answer_prob"],
                    answer_label=answer_label,
                    slot_preds=result["slot_values"] if result["slot_values"] is not None
                        else torch.zeros(input_ids.shape[0], cfg.n_slots, device=device),
                    slot_labels=slot_labels if cfg.use_slots else None,
                    rules=answer_rules,
                )
            total_loss += result["loss_breakdown"].get("L_total", 0.0)
            total_acc += m["accuracy"]
            n_batches += 1

        train_m = {
            "loss": total_loss / n_batches,
            "accuracy": total_acc / n_batches,
        }

        # ── Evaluate ────────────────────────────────────────────
        val_m = _evaluate(model, val_loader, answer_rules, cfg, device)

        history["train"].append(train_m)
        history["val"].append(val_m)

        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Ep {epoch+1:>3}: TrAcc={train_m['accuracy']:.3f}  "
                  f"VlAcc={val_m['accuracy']:.3f}")

        if cfg.patience > 0 and patience_counter >= cfg.patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    return model, history


@torch.no_grad()
def _evaluate(model, loader, answer_rules, cfg, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_acc = 0.0
    total_lfid = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        answer_label = batch["answer_label"].to(device)
        slot_labels = batch["slot_labels"].to(device)

        result = model(input_ids=input_ids, current_epoch=999)
        m = compute_metrics(
            answer_prob=result["answer_prob"],
            answer_label=answer_label,
            slot_preds=result["slot_values"] if result["slot_values"] is not None
                else torch.zeros(input_ids.shape[0], cfg.n_slots, device=device),
            slot_labels=slot_labels if cfg.use_slots else None,
            rules=answer_rules,
        )
        total_acc += m["accuracy"]
        total_lfid += m["logic_fidelity"]
        n_batches += 1

    return {
        "accuracy": total_acc / n_batches,
        "logic_fidelity": total_lfid / n_batches,
    }


@torch.no_grad()
def _evaluate_per_depth(model, val_loader, device):
    """
    Evaluate model accuracy per proof depth.

    Returns dict mapping depth → {"accuracy": float, "count": int}
    """
    model.eval()
    depth_correct = {}   # depth → correct count
    depth_total = {}     # depth → total count

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        answer_label = batch["answer_label"].to(device)

        # proof_depth may not be present in all batches
        proof_depth = batch.get("proof_depth", None)
        if proof_depth is None:
            continue

        result = model(input_ids=input_ids, current_epoch=999)
        preds = (result["answer_prob"] > 0.5).float()
        correct = (preds == answer_label).cpu()

        for i in range(len(proof_depth)):
            d = int(proof_depth[i].item())
            depth_correct[d] = depth_correct.get(d, 0) + int(correct[i].item())
            depth_total[d] = depth_total.get(d, 0) + 1

    results = {}
    for d in sorted(depth_total.keys()):
        results[d] = {
            "accuracy": depth_correct[d] / depth_total[d],
            "count": depth_total[d],
        }
    return results


@torch.no_grad()
def _collect_firing_orders(model, val_loader, device):
    """
    Collect firing orders and proof depths from entire validation set.

    Returns (firing_orders, proof_depths) tensors or (None, None).
    """
    model.eval()
    all_fo = []
    all_pd = []

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        proof_depth = batch.get("proof_depth", None)
        if proof_depth is None:
            continue

        result = model(input_ids=input_ids, current_epoch=999)
        if result["firing_order"] is not None:
            all_fo.append(result["firing_order"].cpu())
            all_pd.append(proof_depth)

    if not all_fo:
        return None, None

    return torch.cat(all_fo, dim=0), torch.cat(all_pd, dim=0)


# ── Experiment Suites ───────────────────────────────────────────────

def run_ablation(
    data_dir: str,
    output_dir: str = "results/ablation",
    device: torch.device = None,
    **train_kwargs,
) -> list[dict]:
    """
    Run 4-way ablation: base, slots_only, loss_only, full.

    Returns list of result dicts with variant, val accuracy, etc.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    variants = ["base", "slots_only", "loss_only", "full"]
    results = []

    for variant in variants:
        print(f"\n{'='*50}")
        print(f"  ABLATION: {variant.upper()}")
        print(f"{'='*50}")

        cfg = NeSyMambaConfig(
            variant=variant,
            d_model=train_kwargs.get("d_model", 64),
            n_layers=train_kwargs.get("n_layers", 2),
            n_slots=train_kwargs.get("n_slots", 7),
            lr=train_kwargs.get("lr", 1e-2),
            epochs=train_kwargs.get("epochs", 20),
            batch_size=train_kwargs.get("batch_size", 128),
            max_seq_len=train_kwargs.get("max_seq_len", 128),
            dataset="proofwriter",
            data_dir=data_dir,
            patience=train_kwargs.get("patience", 20),
            lambda_slot=train_kwargs.get("lambda_slot", 0.0),
            lambda_rule=train_kwargs.get("lambda_rule", 0.0),
            lambda_ortho=train_kwargs.get("lambda_ortho", 0.0),
            grad_clip=train_kwargs.get("grad_clip", 1.0),
            lr_schedule="constant",
        )

        # Build dataloaders
        train_loader, val_loader = get_dataloaders(
            dataset_name="proofwriter",
            batch_size=cfg.batch_size,
            data_dir=data_dir,
            encoding="symbolic",
            n_slots=cfg.n_slots,
            max_seq_len=cfg.max_seq_len,
            max_depth=train_kwargs.get("max_depth", 5),
            max_examples=train_kwargs.get("max_examples", 20000),
        )

        base_ds = train_loader.dataset
        if hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        cfg.vocab_size = len(base_ds.vocab)
        answer_rules = SymbolicProofWriterDataset.get_answer_rules()

        t0 = time.time()
        model, history = _build_model_and_train(
            cfg, train_loader, val_loader, answer_rules, device
        )
        elapsed = time.time() - t0

        final_val = history["val"][-1] if history["val"] else {"accuracy": 0}

        result = {
            "variant": variant,
            "final_val": final_val,
            "epochs_trained": len(history["train"]),
            "train_time_s": round(elapsed, 1),
            "config": cfg.to_dict(),
        }
        results.append(result)

        print(f"  → VlAcc={final_val['accuracy']:.4f}  "
              f"({elapsed:.0f}s, {len(history['train'])} epochs)")

    # Save results
    with open(os.path.join(output_dir, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        plot_ablation_comparison(
            results,
            save_path=os.path.join(output_dir, "ablation_chart.png"),
        )
    except Exception as e:
        print(f"  Warning: plotting failed: {e}")

    return results


def run_compositional_gen(
    data_dir: str,
    train_max_depth: int = 1,
    test_depths: list[int] | None = None,
    output_dir: str = "results/comp_gen",
    device: torch.device = None,
    **train_kwargs,
) -> dict:
    """
    Compositional generalisation: train on depth ≤ D, test on depth > D.

    Tests whether the model can chain reasoning steps it hasn't seen
    during training — a key claim for NeSy architectures.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if test_depths is None:
        test_depths = [0, 1, 2, 3]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  COMPOSITIONAL GENERALISATION")
    print(f"  Train max depth: {train_max_depth}")
    print(f"{'='*50}")

    cfg = NeSyMambaConfig(
        variant="full",
        d_model=train_kwargs.get("d_model", 64),
        n_layers=train_kwargs.get("n_layers", 2),
        n_slots=train_kwargs.get("n_slots", 7),
        lr=train_kwargs.get("lr", 1e-2),
        epochs=train_kwargs.get("epochs", 20),
        batch_size=train_kwargs.get("batch_size", 128),
        max_seq_len=train_kwargs.get("max_seq_len", 128),
        dataset="proofwriter",
        data_dir=data_dir,
        patience=train_kwargs.get("patience", 20),
        lambda_slot=0.0,
        lambda_rule=0.0,
        lambda_ortho=0.0,
        grad_clip=train_kwargs.get("grad_clip", 1.0),
        lr_schedule="constant",
    )

    # Train on limited depth
    train_loader, val_loader = get_dataloaders(
        dataset_name="proofwriter",
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        encoding="symbolic",
        n_slots=cfg.n_slots,
        max_seq_len=cfg.max_seq_len,
        max_depth=train_max_depth,
        max_examples=train_kwargs.get("max_examples", 20000),
    )

    base_ds = train_loader.dataset
    if hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset
    cfg.vocab_size = len(base_ds.vocab)
    answer_rules = SymbolicProofWriterDataset.get_answer_rules()

    model, history = _build_model_and_train(
        cfg, train_loader, val_loader, answer_rules, device
    )

    # Now evaluate on ALL depths (including unseen ones)
    # Need a full-depth validation loader
    _, full_val_loader = get_dataloaders(
        dataset_name="proofwriter",
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        encoding="symbolic",
        n_slots=cfg.n_slots,
        max_seq_len=cfg.max_seq_len,
        max_depth=max(test_depths),
        max_examples=0,  # all validation data
    )

    depth_results = _evaluate_per_depth(model, full_val_loader, device)

    result = {
        "train_max_depth": train_max_depth,
        "test_depths": test_depths,
        "depth_results": {str(k): v for k, v in depth_results.items()},
        "train_epochs": len(history["train"]),
    }

    with open(os.path.join(output_dir, "comp_gen_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    try:
        plot_per_depth_accuracy(
            depth_results,
            title=f"Compositional Gen (train ≤ depth {train_max_depth})",
            save_path=os.path.join(output_dir, "comp_gen_chart.png"),
        )
    except Exception as e:
        print(f"  Warning: plotting failed: {e}")

    # Print results
    for d in sorted(depth_results.keys()):
        seen = "seen" if d <= train_max_depth else "UNSEEN"
        dr = depth_results[d]
        print(f"  Depth {d} ({seen}): acc={dr['accuracy']:.4f}  n={dr['count']}")

    return result


def run_robustness(
    data_dir: str,
    output_dir: str = "results/robustness",
    device: torch.device = None,
    **train_kwargs,
) -> dict:
    """
    Fact-order robustness: compare accuracy with original vs shuffled premise order.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  FACT-ORDER ROBUSTNESS TEST")
    print(f"{'='*50}")

    results = {}
    for shuffle in [False, True]:
        label = "shuffled" if shuffle else "original"
        print(f"\n  → Training with {label} order...")

        cfg = NeSyMambaConfig(
            variant="full",
            d_model=train_kwargs.get("d_model", 64),
            n_layers=train_kwargs.get("n_layers", 2),
            n_slots=train_kwargs.get("n_slots", 7),
            lr=train_kwargs.get("lr", 1e-2),
            epochs=train_kwargs.get("epochs", 20),
            batch_size=train_kwargs.get("batch_size", 128),
            max_seq_len=train_kwargs.get("max_seq_len", 128),
            dataset="proofwriter",
            data_dir=data_dir,
            patience=train_kwargs.get("patience", 20),
            lambda_slot=0.0,
            lambda_rule=0.0,
            lambda_ortho=0.0,
            grad_clip=train_kwargs.get("grad_clip", 1.0),
            lr_schedule="constant",
        )

        pw_kwargs = dict(
            n_slots=cfg.n_slots,
            max_seq_len=cfg.max_seq_len,
            max_depth=train_kwargs.get("max_depth", 5),
        )
        if shuffle:
            pw_kwargs["shuffle_facts"] = True

        train_loader, val_loader = get_dataloaders(
            dataset_name="proofwriter",
            batch_size=cfg.batch_size,
            data_dir=data_dir,
            encoding="symbolic",
            max_examples=train_kwargs.get("max_examples", 20000),
            **pw_kwargs,
        )

        base_ds = train_loader.dataset
        if hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        cfg.vocab_size = len(base_ds.vocab)
        answer_rules = SymbolicProofWriterDataset.get_answer_rules()

        model, history = _build_model_and_train(
            cfg, train_loader, val_loader, answer_rules, device
        )

        final_val = history["val"][-1] if history["val"] else {"accuracy": 0}
        results[label] = {
            "accuracy": final_val["accuracy"],
            "epochs": len(history["train"]),
        }
        print(f"    VlAcc={final_val['accuracy']:.4f}")

    # Compute degradation
    orig_acc = results["original"]["accuracy"]
    shuf_acc = results["shuffled"]["accuracy"]
    degradation = orig_acc - shuf_acc

    results["degradation"] = degradation
    results["degradation_pct"] = (degradation / max(orig_acc, 1e-8)) * 100

    print(f"\n  Degradation: {degradation:+.4f} ({results['degradation_pct']:+.1f}%)")

    with open(os.path.join(output_dir, "robustness_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_firing_order_analysis(
    data_dir: str,
    output_dir: str = "results/firing_order",
    device: torch.device = None,
    checkpoint_path: str | None = None,
    **train_kwargs,
) -> dict:
    """
    Analyse correlation between slot firing timestep and proof depth.

    If checkpoint_path is given, loads a pretrained model.
    Otherwise trains a new model first.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  FIRING ORDER VS PROOF DEPTH ANALYSIS")
    print(f"{'='*50}")

    cfg = NeSyMambaConfig(
        variant="full",
        d_model=train_kwargs.get("d_model", 64),
        n_layers=train_kwargs.get("n_layers", 2),
        n_slots=train_kwargs.get("n_slots", 7),
        lr=train_kwargs.get("lr", 1e-2),
        epochs=train_kwargs.get("epochs", 20),
        batch_size=train_kwargs.get("batch_size", 128),
        max_seq_len=train_kwargs.get("max_seq_len", 128),
        dataset="proofwriter",
        data_dir=data_dir,
        patience=train_kwargs.get("patience", 20),
        lambda_slot=0.0,
        lambda_rule=0.0,
        lambda_ortho=0.0,
        grad_clip=train_kwargs.get("grad_clip", 1.0),
        lr_schedule="constant",
    )

    train_loader, val_loader = get_dataloaders(
        dataset_name="proofwriter",
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        encoding="symbolic",
        n_slots=cfg.n_slots,
        max_seq_len=cfg.max_seq_len,
        max_depth=train_kwargs.get("max_depth", 5),
        max_examples=train_kwargs.get("max_examples", 20000),
    )

    base_ds = train_loader.dataset
    if hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset
    cfg.vocab_size = len(base_ds.vocab)
    answer_rules = SymbolicProofWriterDataset.get_answer_rules()

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = NeSyMamba(cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        model, _ = _build_model_and_train(
            cfg, train_loader, val_loader, answer_rules, device
        )

    # Collect firing orders
    firing_orders, proof_depths = _collect_firing_orders(model, val_loader, device)

    if firing_orders is None:
        print("  No proof depths available — skipping analysis.")
        return {}

    # Compute correlation
    stats = compute_firing_depth_correlation(firing_orders, proof_depths)
    print(f"  Spearman ρ = {stats['spearman_r']:.4f}  (p={stats['spearman_p']:.4e})")
    print(f"  N examples = {stats['n_examples']}")
    for d, mf in stats["depth_to_mean_firing"].items():
        print(f"    Depth {d}: mean firing time = {mf:.1f}")

    # Plot
    try:
        plot_firing_order_vs_depth(
            firing_orders, proof_depths,
            title="Slot Firing Time vs Proof Depth",
            save_path=os.path.join(output_dir, "firing_vs_depth.png"),
        )
    except Exception as e:
        print(f"  Warning: plotting failed: {e}")

    with open(os.path.join(output_dir, "firing_order_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def run_per_depth_eval(
    data_dir: str,
    output_dir: str = "results/per_depth",
    device: torch.device = None,
    checkpoint_path: str | None = None,
    **train_kwargs,
) -> dict:
    """
    Per-depth accuracy breakdown on validation set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  PER-DEPTH ACCURACY BREAKDOWN")
    print(f"{'='*50}")

    cfg = NeSyMambaConfig(
        variant="full",
        d_model=train_kwargs.get("d_model", 64),
        n_layers=train_kwargs.get("n_layers", 2),
        n_slots=train_kwargs.get("n_slots", 7),
        lr=train_kwargs.get("lr", 1e-2),
        epochs=train_kwargs.get("epochs", 20),
        batch_size=train_kwargs.get("batch_size", 128),
        max_seq_len=train_kwargs.get("max_seq_len", 128),
        dataset="proofwriter",
        data_dir=data_dir,
        patience=train_kwargs.get("patience", 20),
        lambda_slot=0.0,
        lambda_rule=0.0,
        lambda_ortho=0.0,
        grad_clip=train_kwargs.get("grad_clip", 1.0),
        lr_schedule="constant",
    )

    train_loader, val_loader = get_dataloaders(
        dataset_name="proofwriter",
        batch_size=cfg.batch_size,
        data_dir=data_dir,
        encoding="symbolic",
        n_slots=cfg.n_slots,
        max_seq_len=cfg.max_seq_len,
        max_depth=train_kwargs.get("max_depth", 5),
        max_examples=train_kwargs.get("max_examples", 20000),
    )

    base_ds = train_loader.dataset
    if hasattr(base_ds, "dataset"):
        base_ds = base_ds.dataset
    cfg.vocab_size = len(base_ds.vocab)
    answer_rules = SymbolicProofWriterDataset.get_answer_rules()

    if checkpoint_path and os.path.exists(checkpoint_path):
        model = NeSyMamba(cfg).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {checkpoint_path}")
    else:
        model, _ = _build_model_and_train(
            cfg, train_loader, val_loader, answer_rules, device
        )

    depth_results = _evaluate_per_depth(model, val_loader, device)

    for d in sorted(depth_results.keys()):
        dr = depth_results[d]
        print(f"  Depth {d}: acc={dr['accuracy']:.4f}  n={dr['count']}")

    try:
        plot_per_depth_accuracy(
            depth_results,
            save_path=os.path.join(output_dir, "per_depth_accuracy.png"),
        )
    except Exception as e:
        print(f"  Warning: plotting failed: {e}")

    with open(os.path.join(output_dir, "per_depth_results.json"), "w") as f:
        json.dump({str(k): v for k, v in depth_results.items()}, f, indent=2)

    return depth_results


# ── CLI Entry Point ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NeSy Mamba Experiment Runner")
    p.add_argument("--suite", type=str, required=True,
                   choices=["ablation", "comp_gen", "robustness",
                            "firing_order", "per_depth", "all"],
                   help="Which experiment suite to run")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Root data dir containing proofwriter/ folder")
    p.add_argument("--output_dir", type=str, default="results",
                   help="Output directory for results and plots")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_slots", type=int, default=7)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--max_examples", type=int, default=20000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to pretrained checkpoint (for eval-only suites)")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    kwargs = {
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_slots": args.n_slots,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "max_depth": args.max_depth,
        "max_examples": args.max_examples,
        "patience": args.patience,
    }

    suites_to_run = []
    if args.suite == "all":
        suites_to_run = ["ablation", "per_depth", "firing_order",
                         "comp_gen", "robustness"]
    else:
        suites_to_run = [args.suite]

    all_results = {}

    for suite in suites_to_run:
        if suite == "ablation":
            all_results["ablation"] = run_ablation(
                args.data_dir,
                os.path.join(args.output_dir, "ablation"),
                device, **kwargs,
            )
        elif suite == "per_depth":
            all_results["per_depth"] = run_per_depth_eval(
                args.data_dir,
                os.path.join(args.output_dir, "per_depth"),
                device, checkpoint_path=args.checkpoint, **kwargs,
            )
        elif suite == "firing_order":
            all_results["firing_order"] = run_firing_order_analysis(
                args.data_dir,
                os.path.join(args.output_dir, "firing_order"),
                device, checkpoint_path=args.checkpoint, **kwargs,
            )
        elif suite == "comp_gen":
            all_results["comp_gen"] = run_compositional_gen(
                args.data_dir,
                train_max_depth=1,
                output_dir=os.path.join(args.output_dir, "comp_gen"),
                device=device, **kwargs,
            )
        elif suite == "robustness":
            all_results["robustness"] = run_robustness(
                args.data_dir,
                os.path.join(args.output_dir, "robustness"),
                device, **kwargs,
            )

    print(f"\n{'='*50}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Results saved to: {args.output_dir}/")
    print(f"{'='*50}")

    return all_results


if __name__ == "__main__":
    main()
