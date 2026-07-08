"""
Training script for Slot-Gated NeSy Mamba.

Supports 4 ablation variants:
  base       — vanilla Mamba, no slots, no logic loss
  slots_only — Mamba + slots, no logic loss
  loss_only  — Mamba + logic loss on outputs, no explicit slots
  full       — Mamba + slots + logic losses (complete NeSy model)

Usage:
  python -m nesy_mamba.train --variant full --epochs 50
"""

import argparse
import math
import os
import sys
import torch
import torch.nn as nn
from .config import NeSyMambaConfig
from .nesy_mamba import NeSyMamba
from .data_utils import (
    get_dataloaders,
    SyntheticRulesDataset,
    ProofWriterDataset,
    SymbolicProofWriterDataset,
    CLUTRRDataset,
    load_glove_embeddings,
    collate_fn,
)
from .metrics import compute_metrics
from .proof_parser import extract_dynamic_rules, parse_proof_batch


# ── Dataset-specific defaults ────────────────────────────────────────
DATASET_DEFAULTS = {
    "synthetic": dict(
        vocab_size=50, max_seq_len=32, n_slots=7,
        lr=1e-3, epochs=50, batch_size=32,
    ),
    "proofwriter": dict(
        vocab_size=None,   # derived from data
        max_seq_len=128,
        n_slots=7,
        lr=5e-3,
        epochs=50,
        batch_size=64,
    ),
    "clutrr": dict(
        vocab_size=None,
        max_seq_len=256,
        n_slots=8,
        lr=3e-4,
        epochs=30,
        batch_size=16,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Train NeSy Mamba")
    p.add_argument("--variant", type=str, default="full",
                   choices=["base", "slots_only", "loss_only", "full"])
    p.add_argument("--dataset", type=str, default="synthetic",
                   choices=["synthetic", "proofwriter", "clutrr"])
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root data dir containing proofwriter/ or clutrr/ folders")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_slots", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--slot_warmup", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_depth", type=int, default=5,
                   help="Max proof depth (ProofWriter only)")
    p.add_argument("--pw_subset", type=str, default=None,
                   help="ProofWriter subdirectory, e.g. CWA/depth-3ext-NatLang")
    p.add_argument("--max_examples", type=int, default=20000,
                   help="Max training examples (subsample for speed). 0=all.")
    p.add_argument("--lambda_slot", type=float, default=None,
                   help="Slot supervision loss weight (default: 0.2 for real data, 1.0 for synthetic)")
    p.add_argument("--lambda_rule", type=float, default=None,
                   help="Answer-consistency loss weight")
    p.add_argument("--lambda_ortho", type=float, default=None,
                   help="Orthogonality loss weight")
    p.add_argument("--lambda_entropy", type=float, default=None,
                   help="Slot entropy/anti-collapse loss weight (default: 0.1)")
    p.add_argument("--glove_path", type=str, default=None,
                   help="Path to GloVe file (e.g. glove.6B.100d.txt). "
                        "Strongly recommended for ProofWriter/CLUTRR.")
    p.add_argument("--freeze_emb", action="store_true",
                   help="Freeze pretrained embeddings (don't finetune)")
    p.add_argument("--encoding", type=str, default="text",
                   choices=["text", "symbolic"],
                   help="Input encoding: 'text' (raw NL) or 'symbolic' "
                        "(structured tokens from triple/rule representations). "
                        "Only affects ProofWriter.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--shuffle_facts", action="store_true",
                   help="Randomise premise order (order sensitivity test)")
    p.add_argument("--slot_competition", action="store_true",
                   help="Enable softmax competition among slot candidates")
    p.add_argument("--slot_temperature", type=float, default=0.5,
                   help="Temperature for slot competition softmax (lower = sharper)")
    p.add_argument("--coupled_slots", action="store_true",
                   help="Use full K×K slot recurrence instead of diagonal (ablation)")
    p.add_argument("--dynamic_rules", action="store_true",
                   help="Use proof-conditioned per-example rules for L_rule")
    p.add_argument("--curriculum", type=str, default=None,
                   help="Curriculum schedule as comma-separated max depths per stage, "
                        "e.g. '0,1,2,3' means stage1=depth0, stage2=depth≤1, etc. "
                        "Epochs are split evenly across stages.")
    p.add_argument("--slot_bias_init", type=float, default=None,
                   help="Initial bias for slot gates (default: -3.0). "
                        "Higher = slots start more active.")
    p.add_argument("--slot_recurrence_init", type=float, default=None,
                   help="Diagonal recurrence init (default: -0.1). "
                        "Positive = self-exciting, negative = self-inhibiting.")
    p.add_argument("--weight_decay", type=float, default=None,
                   help="AdamW weight decay (default: from config = 0.01)")
    p.add_argument("--slot_gate_mode", type=str, default=None,
                   choices=["monotonic", "ema"],
                   help="Slot gate mode: 'monotonic' (max) or 'ema' (learnable EMA)")
    p.add_argument("--slot_ema_alpha_init", type=float, default=None,
                   help="Initial logit for EMA alpha (default: 2.0, sigmoid≈0.88)")
    p.add_argument("--slot_ema_dynamic_alpha", type=str, default=None,
                   choices=["true", "false"],
                   help="Input-dependent alpha: α_k(t)=σ(W_α·h_t). Default: true")
    p.add_argument("--slot_ws_gain", type=float, default=None,
                   help="Xavier gain for W_s projection (default: 0.1). "
                        "Higher = stronger input signal for slots.")
    p.add_argument("--slot_dropout", type=float, default=None,
                   help="Dropout rate on slot activations (default: 0.0). "
                        "Prevents over-reliance on single slots.")
    # v10: Slot routing (Top-K hard routing)
    p.add_argument("--slot_routing", action="store_true",
                   help="Enable Top-K hard slot routing (v10, MoE-style)")
    p.add_argument("--slot_routing_top_k", type=int, default=None,
                   help="Number of slots each token routes to (default: 2)")
    p.add_argument("--lambda_balance", type=float, default=None,
                   help="Load-balancing loss weight for slot router (default: 0.01)")
    p.add_argument("--slot_ortho_init", action="store_true",
                   help="Orthogonal initialization for slot projections (v10)")
    p.add_argument("--slot_label_mode", type=str, default="type",
                   choices=["type", "index"],
                   help="Slot label mode: 'type' (rule-type-based, globally consistent) "
                        "or 'index' (legacy position-based). Default: type.")
    return p.parse_args()


@torch.no_grad()
def _compute_slot_diagnostics(model, loader, device) -> dict:
    """
    Compute slot health diagnostics over one pass of the data loader.

    Returns dict with:
        gate_mean:           mean slot activation across all examples/slots
        gate_std:            std of slot activations
        frac_slots_above_05: fraction of (example, slot) pairs > 0.5
        slot_entropy:        mean per-example entropy of slot distribution
        frac_collapsed:      fraction of slots that are always < 0.1 or always > 0.9
    """
    model.eval()
    all_vals = []

    # Sample up to 500 examples for speed
    n_seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        result = model(input_ids=input_ids, current_epoch=999)
        if result["slot_values"] is not None:
            all_vals.append(result["slot_values"].cpu())
        n_seen += input_ids.shape[0]
        if n_seen >= 500:
            break

    model.train()

    if not all_vals:
        return {}

    vals = torch.cat(all_vals, dim=0)  # (N, K)
    N, K = vals.shape

    gate_mean = vals.mean().item()
    gate_std = vals.std().item()
    frac_above = (vals > 0.5).float().mean().item()

    # Per-example entropy: H = -Σ p_i log(p_i + eps) for normalized slots
    eps = 1e-8
    p = vals / (vals.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=-1).mean().item()

    # Collapsed slots: per-slot mean across examples
    slot_means = vals.mean(dim=0)  # (K,)
    collapsed = ((slot_means < 0.1) | (slot_means > 0.9)).float().mean().item()

    return {
        "gate_mean": round(gate_mean, 4),
        "gate_std": round(gate_std, 4),
        "frac_slots_above_05": round(frac_above, 4),
        "slot_entropy": round(entropy, 4),
        "frac_collapsed": round(collapsed, 4),
    }


def train_one_epoch(
    model: NeSyMamba,
    loader,
    optimizer,
    answer_rules,
    epoch: int,
    device: torch.device,
    grad_clip: float,
) -> dict:
    """Train for one epoch, return average metrics."""
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_acc = 0.0
    total_lfid = 0.0
    total_sfid = 0.0
    total_grad_norm = 0.0
    n_batches = 0

    import time as _time
    total_batches = len(loader)
    t0 = _time.time()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        answer_label = batch["answer_label"].to(device)
        slot_labels = batch["slot_labels"].to(device)

        # Extract per-example dynamic rules from proof strings (if available)
        per_example_rules = None
        proof_strs = batch.get("proof_str", None)
        if proof_strs is not None:
            proof_trees = parse_proof_batch(proof_strs)
            per_example_rules = extract_dynamic_rules(proof_trees, n_slots=model.cfg.n_slots)

        optimizer.zero_grad()

        result = model(
            input_ids=input_ids,
            answer_labels=answer_label,
            slot_labels=slot_labels if model.cfg.use_slots else None,
            rules=answer_rules,
            per_example_rules=per_example_rules,
            current_epoch=epoch,
        )

        loss = result["total_loss"]
        loss.backward()

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        total_grad_norm += grad_norm.item()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            m = compute_metrics(
                answer_prob=result["answer_prob"],
                answer_label=answer_label,
                slot_preds=result["slot_values"] if result["slot_values"] is not None
                    else torch.zeros(input_ids.shape[0], model.cfg.n_slots, device=device),
                slot_labels=slot_labels if model.cfg.use_slots else None,
                rules=answer_rules,
            )

        total_loss += result["loss_breakdown"].get("L_total", 0.0)
        total_task_loss += result["loss_breakdown"].get("L_task", 0.0)
        total_acc += m["accuracy"]
        total_lfid += m["logic_fidelity"]
        total_sfid += m.get("slot_fidelity", 0.0)
        n_batches += 1

        # Progress every 25 batches
        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
            elapsed = _time.time() - t0
            eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
            print(f"  [{batch_idx+1}/{total_batches}] "
                  f"Ltask={total_task_loss/n_batches:.4f}  "
                  f"Ltot={total_loss/n_batches:.4f}  "
                  f"acc={total_acc/n_batches:.3f}  "
                  f"ETA={eta:.0f}s", end="\r")

    print()  # newline after progress
    metrics = {
        "loss": total_loss / n_batches,
        "task_loss": total_task_loss / n_batches,
        "accuracy": total_acc / n_batches,
        "logic_fidelity": total_lfid / n_batches,
        "slot_fidelity": total_sfid / n_batches,
        "grad_norm": total_grad_norm / n_batches,
    }

    # ── Slot health diagnostics ─────────────────────────────────
    if model.cfg.use_slots and model.slot_gate is not None:
        slot_diag = _compute_slot_diagnostics(model, loader, device)
        metrics.update(slot_diag)

    return metrics


@torch.no_grad()
def evaluate(
    model: NeSyMamba,
    loader,
    answer_rules,
    device: torch.device,
) -> dict:
    """Evaluate on validation set with per-depth breakdown."""
    model.eval()
    total_acc = 0.0
    total_lfid = 0.0
    total_sfid = 0.0
    n_batches = 0

    # Per-depth tracking
    from collections import defaultdict
    depth_correct = defaultdict(int)
    depth_total = defaultdict(int)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        answer_label = batch["answer_label"].to(device)
        slot_labels = batch["slot_labels"].to(device)

        result = model(input_ids=input_ids, current_epoch=999)

        m = compute_metrics(
            answer_prob=result["answer_prob"],
            answer_label=answer_label,
            slot_preds=result["slot_values"] if result["slot_values"] is not None
                else torch.zeros(input_ids.shape[0], model.cfg.n_slots, device=device),
            slot_labels=slot_labels if model.cfg.use_slots else None,
            rules=answer_rules,
        )

        total_acc += m["accuracy"]
        total_lfid += m["logic_fidelity"]
        total_sfid += m.get("slot_fidelity", 0.0)
        n_batches += 1

        # Per-depth accuracy
        if "proof_depth" in batch:
            depths = batch["proof_depth"]
            preds = (result["answer_prob"] > 0.5).float()
            correct = (preds == answer_label).cpu()
            for i in range(len(depths)):
                d = depths[i].item()
                depth_correct[d] += correct[i].item()
                depth_total[d] += 1

    metrics = {
        "accuracy": total_acc / n_batches,
        "logic_fidelity": total_lfid / n_batches,
        "slot_fidelity": total_sfid / n_batches,
    }

    # Add per-depth accuracy
    for d in sorted(depth_total.keys()):
        if depth_total[d] > 0:
            metrics[f"acc_d{d}"] = depth_correct[d] / depth_total[d]

    return metrics


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Resolve defaults per dataset ────────────────────────────────
    ds_defs = DATASET_DEFAULTS[args.dataset]
    epochs     = args.epochs     or ds_defs["epochs"]
    lr         = args.lr         or ds_defs["lr"]
    batch_size = args.batch_size or ds_defs["batch_size"]
    n_slots    = args.n_slots    or ds_defs["n_slots"]
    max_seq_len = args.max_seq_len or ds_defs["max_seq_len"]

    # Symbolic encoding uses shorter sequences by default
    if args.encoding == "symbolic" and args.max_seq_len is None:
        max_seq_len = 64

    # Resolve data_dir: default to <workspace>/nesy_mamba/data
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")

    # ── Build data loaders ──────────────────────────────────────────
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "synthetic":
        vocab_size = ds_defs["vocab_size"]
        train_loader, val_loader = get_dataloaders(
            dataset_name="synthetic",
            batch_size=batch_size,
            n_train=2000,
            n_val=500,
            seq_len=max_seq_len,
            vocab_size=vocab_size,
            n_slots=n_slots,
        )
        answer_rules = SyntheticRulesDataset.get_answer_rules()
        rule_names = SyntheticRulesDataset.get_rule_names()

    elif args.dataset == "proofwriter":
        pw_kwargs = dict(
            n_slots=n_slots,
            max_seq_len=max_seq_len,
            max_depth=args.max_depth,
            slot_label_mode=args.slot_label_mode,
        )
        if args.shuffle_facts:
            pw_kwargs["shuffle_facts"] = True
        if args.pw_subset:
            pw_kwargs["subset"] = args.pw_subset
        train_loader, val_loader = get_dataloaders(
            dataset_name="proofwriter",
            batch_size=batch_size,
            data_dir=data_dir,
            max_examples=args.max_examples,
            encoding=args.encoding,
            **pw_kwargs,
        )
        # Derive vocab_size from the training set's vocab
        train_ds = train_loader.dataset
        # Handle Subset wrapping
        base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        vocab_size = len(base_ds.vocab)
        if args.encoding == "symbolic":
            answer_rules = SymbolicProofWriterDataset.get_answer_rules()
            rule_names = SymbolicProofWriterDataset.get_rule_names()
        else:
            answer_rules = ProofWriterDataset.get_answer_rules()
            rule_names = ProofWriterDataset.get_rule_names()

    elif args.dataset == "clutrr":
        cl_kwargs = dict(n_slots=n_slots, max_seq_len=max_seq_len)
        train_loader, val_loader = get_dataloaders(
            dataset_name="clutrr",
            batch_size=batch_size,
            data_dir=data_dir,
            **cl_kwargs,
        )
        train_ds = train_loader.dataset
        base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
        vocab_size = len(base_ds.vocab)
        answer_rules = CLUTRRDataset.get_answer_rules()
        rule_names = CLUTRRDataset.get_rule_names()

    # ── Auto-compute pos_weight from class balance ──────────────
    pos_weight_val = 1.0
    if args.dataset in ("proofwriter", "clutrr"):
        train_ds = train_loader.dataset
        n_total = len(train_ds)
        n_pos = 0
        for i in range(n_total):
            if train_ds[i]["answer_label"].item() > 0.5:
                n_pos += 1
        n_neg = n_total - n_pos
        if n_pos > 0 and n_neg > 0:
            pos_weight_val = n_neg / n_pos
        print(f"  Class balance: {n_pos} True / {n_neg} False ({100*n_pos/n_total:.1f}%)")
        print(f"  pos_weight (auto): {pos_weight_val:.4f}")

    # Config
    cfg = NeSyMambaConfig(
        variant=args.variant,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_slots=n_slots,
        slot_warmup_epochs=args.slot_warmup,
        grad_clip=args.grad_clip,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dataset=args.dataset,
        data_dir=data_dir,
        slot_competition=args.slot_competition,
        slot_temperature=args.slot_temperature,
        coupled_slots=args.coupled_slots,
        pos_weight=pos_weight_val,
    )

    # Override slot init values if provided via CLI
    if args.slot_bias_init is not None:
        cfg.slot_bias_init = args.slot_bias_init
    if args.slot_recurrence_init is not None:
        cfg.slot_recurrence_init = args.slot_recurrence_init
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.slot_gate_mode is not None:
        cfg.slot_gate_mode = args.slot_gate_mode
    if args.slot_ema_alpha_init is not None:
        cfg.slot_ema_alpha_init = args.slot_ema_alpha_init
    if hasattr(args, 'slot_ema_dynamic_alpha') and args.slot_ema_dynamic_alpha is not None:
        cfg.slot_ema_dynamic_alpha = args.slot_ema_dynamic_alpha.lower() == "true"
    if args.slot_ws_gain is not None:
        cfg.slot_ws_gain = args.slot_ws_gain
    if args.slot_dropout is not None:
        cfg.slot_dropout = args.slot_dropout
    # v10: Slot routing
    if getattr(args, 'slot_routing', False):
        cfg.slot_routing = True
    if getattr(args, 'slot_routing_top_k', None) is not None:
        cfg.slot_routing_top_k = args.slot_routing_top_k
    if getattr(args, 'lambda_balance', None) is not None:
        cfg.lambda_balance = args.lambda_balance
    if getattr(args, 'slot_ortho_init', False):
        cfg.slot_ortho_init = True

    # Override logic loss weights for real datasets
    if args.dataset != "synthetic":
        cfg.lambda_slot  = args.lambda_slot  if args.lambda_slot  is not None else 0.2
        # L_rule disabled for ProofWriter: static answer_rules don't match
        # actual per-theory rule structure.  L_slot provides the main
        # neuro-symbolic signal.
        cfg.lambda_rule  = args.lambda_rule  if args.lambda_rule  is not None else 0.0
        cfg.lambda_ortho = args.lambda_ortho if args.lambda_ortho is not None else 0.05
        cfg.lambda_entropy = args.lambda_entropy if args.lambda_entropy is not None else 0.1
    else:
        if args.lambda_slot  is not None: cfg.lambda_slot  = args.lambda_slot
        if args.lambda_rule  is not None: cfg.lambda_rule  = args.lambda_rule
        if args.lambda_ortho is not None: cfg.lambda_ortho = args.lambda_ortho
        if args.lambda_entropy is not None: cfg.lambda_entropy = args.lambda_entropy

    print(f"\n{'='*60}")
    enc_label = f" ({args.encoding})" if args.dataset == "proofwriter" else ""
    print(f"  NeSy Mamba -- Variant: {cfg.variant.upper()}")
    print(f"  Dataset: {args.dataset}{enc_label}  |  Vocab: {vocab_size:,}")
    print(f"  Slots: {'YES' if cfg.use_slots else 'NO'}  |  "
          f"Logic Loss: {'YES' if cfg.use_logic_loss else 'NO'}")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
          f"n_slots={cfg.n_slots}, max_seq={max_seq_len}")
    print(f"  Loss weights: L_slot={cfg.lambda_slot}, L_rule={cfg.lambda_rule}, "
          f"L_ortho={cfg.lambda_ortho}")
    print(f"  Slot warmup: {cfg.slot_warmup_epochs} epochs")
    print(f"{'='*60}\n")

    # Model
    model = NeSyMamba(cfg).to(device)

    # ── Pretrained embeddings (GloVe) ───────────────────────────────
    vocab_obj = None
    if args.dataset == "proofwriter":
        base_ds = train_loader.dataset
        if hasattr(base_ds, 'dataset'): base_ds = base_ds.dataset
        vocab_obj = base_ds.vocab
    elif args.dataset == "clutrr":
        base_ds = train_loader.dataset
        if hasattr(base_ds, 'dataset'): base_ds = base_ds.dataset
        vocab_obj = base_ds.vocab

    if args.glove_path and vocab_obj is not None:
        glove_weight = load_glove_embeddings(args.glove_path, vocab_obj, cfg.d_model)
        model.embedding.weight.data.copy_(glove_weight.to(device))
        if args.freeze_emb:
            model.embedding.weight.requires_grad = False
            print("  Embeddings frozen (not finetuned)")
        else:
            print("  Embeddings initialized from GloVe (will finetune)")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_trainable:,} trainable)\n")

    # ── Optimizer with differential LR ──────────────────────────────
    if args.glove_path and not args.freeze_emb:
        # Lower LR for pretrained embeddings, full LR for the rest
        emb_params = list(model.embedding.parameters()) + list(model.pos_embedding.parameters())
        emb_ids = {id(p) for p in emb_params}
        other_params = [p for p in model.parameters() if id(p) not in emb_ids and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": emb_params, "lr": cfg.lr * 0.1},
            {"params": other_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)
        print(f"  Differential LR: emb={cfg.lr*0.1:.1e}, rest={cfg.lr:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    # LR schedule: linear warmup for first 2 epochs, then cosine decay
    warmup_epochs = min(2, cfg.epochs // 5) if cfg.epochs > 4 else 0
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)   # linear warmup
        # cosine decay over remaining epochs
        progress = (epoch - warmup_epochs) / max(cfg.epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Curriculum learning setup ─────────────────────────────────────
    curriculum_sampler = None
    curriculum_schedule = None   # list of (start_epoch, max_depth)

    if args.curriculum and args.dataset == "proofwriter":
        from .data_utils import CurriculumSampler
        # Parse schedule: "0,1,2,3" → 4 stages, evenly split across epochs
        stages = [int(d) for d in args.curriculum.split(",")]
        epochs_per_stage = max(1, epochs // len(stages))
        curriculum_schedule = []
        for i, max_d in enumerate(stages):
            curriculum_schedule.append((i * epochs_per_stage, max_d))
        print(f"  Curriculum schedule: {curriculum_schedule}")
        print(f"    ({epochs_per_stage} epochs per stage)")

        # Build sampler from the base training dataset
        base_train = train_loader.dataset
        curriculum_sampler = CurriculumSampler(base_train)

    # ── Training loop ───────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train": [], "val": []}

    header = (f"{'Ep':>3} | {'LTask':>7} | {'LTot':>7} | {'TrAcc':>6} | {'VlAcc':>6} | "
              f"{'VlLFid':>6} | {'VlSFid':>6} | {'GNorm':>6} | {'LR':>8} | {'Phase':>10}")
    print(header)
    print("-" * len(header))

    for epoch in range(cfg.epochs):
        # ── Curriculum: rebuild train_loader if depth stage changed ──
        if curriculum_sampler is not None and curriculum_schedule is not None:
            # Find the active stage for this epoch
            active_depth = curriculum_schedule[0][1]
            for start_ep, max_d in curriculum_schedule:
                if epoch >= start_ep:
                    active_depth = max_d
            if active_depth != curriculum_sampler.current_depth:
                curriculum_sampler.set_max_depth(active_depth)
                train_subset = curriculum_sampler.get_subset()
                from torch.utils.data import DataLoader as DL
                train_loader = DL(
                    train_subset, batch_size=batch_size, shuffle=True,
                    collate_fn=collate_fn,
                )
                depth_summary = curriculum_sampler.summary()
                print(f"  [Curriculum] depth≤{active_depth}: "
                      f"{curriculum_sampler.n_examples} examples  "
                      f"{depth_summary}")

        # Phase label
        if epoch < cfg.slot_warmup_epochs and cfg.use_logic_loss:
            phase = "warmup"
        elif curriculum_sampler is not None:
            phase = f"curric-d{curriculum_sampler.current_depth}"
        else:
            phase = "full-train"

        train_m = train_one_epoch(
            model, train_loader, optimizer, answer_rules, epoch, device, cfg.grad_clip
        )
        val_m = evaluate(model, val_loader, answer_rules, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        # Save best model
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            best_epoch = epoch
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_m,
                "config": vars(cfg),
            }, os.path.join("checkpoints", f"best_{cfg.variant}.pt"))

        print(
            f"{epoch+1:>3} | {train_m['task_loss']:>7.4f} | "
            f"{train_m['loss']:>7.4f} | "
            f"{train_m['accuracy']:>6.3f} | "
            f"{val_m['accuracy']:>6.3f} | "
            f"{val_m['logic_fidelity']:>6.3f} | "
            f"{val_m.get('slot_fidelity', 0):>6.3f} | "
            f"{train_m.get('grad_norm', 0):>6.2f} | "
            f"{current_lr:>8.6f} | "
            f"{phase:>10}"
        )

        # Per-depth accuracy breakdown
        depth_accs = {k: v for k, v in val_m.items() if k.startswith("acc_d")}
        if depth_accs:
            depth_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(depth_accs.items()))
            print(f"    depths: {depth_str}")

        # Log slot health diagnostics (if computed)
        if "gate_mean" in train_m:
            print(
                f"    slots: mean={train_m['gate_mean']:.3f} "
                f"std={train_m['gate_std']:.3f} "
                f"above0.5={train_m['frac_slots_above_05']:.1%} "
                f"entropy={train_m['slot_entropy']:.3f} "
                f"collapsed={train_m['frac_collapsed']:.1%}"
            )

    # ── Load best model for final evaluation ────────────────────────
    ckpt_path = os.path.join("checkpoints", f"best_{cfg.variant}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\n  Loaded best model from epoch {best_epoch + 1}")

    # ── Evaluation ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION ON VALIDATION SET (best epoch: {best_epoch + 1})")
    print(f"{'='*60}")

    val_m = evaluate(model, val_loader, answer_rules, device)
    print(f"  Accuracy:       {val_m['accuracy']:.4f}")
    print(f"  Logic Fidelity: {val_m['logic_fidelity']:.4f}")
    print(f"  Slot Fidelity:  {val_m.get('slot_fidelity', 0):.4f}")

    # Per-depth accuracy table
    depth_accs = {k: v for k, v in val_m.items() if k.startswith("acc_d")}
    if depth_accs:
        print(f"\n  Per-Depth Accuracy:")
        for k, v in sorted(depth_accs.items()):
            print(f"    {k}: {v:.4f}")

    # ── Self-explanation demo ───────────────────────────────────────
    if cfg.use_slots:
        print(f"\n{'='*60}")
        print("  SELF-EXPLANATION DEMO (first 3 validation examples)")
        print(f"{'='*60}")

        sample_batch = next(iter(val_loader))
        sample_ids = sample_batch["input_ids"][:3].to(device)
        sample_labels = sample_batch["answer_label"][:3]

        explanations = model.explain(sample_ids, rule_names)
        for i, (exp, true_ans) in enumerate(zip(explanations, sample_labels)):
            print(f"\n  Example {i+1}:")
            print(f"    True answer:  {bool(true_ans.item())}")
            print(f"    Predicted:    {exp['answer']} "
                  f"(confidence={exp['confidence']})")
            if exp["trace"]:
                print(f"    Reasoning trace:")
                for step in exp["trace"]:
                    print(f"      Step {step['step']:>3}: "
                          f"{step['rule']} = {step['value']}")
            else:
                print(f"    No slots fired.")

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")

    return history


if __name__ == "__main__":
    main()
