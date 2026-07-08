"""Quick end-to-end test of type-based slot labels through full pipeline."""
import torch
import sys
sys.path.insert(0, ".")

from nesy_mamba.data_utils import get_dataloaders, SymbolicProofWriterDataset, RULE_TYPE_NAMES
from nesy_mamba.config import NeSyMambaConfig
from nesy_mamba.nesy_mamba import NeSyMamba
from nesy_mamba.metrics import compute_metrics

print("=== Testing type-based slot labels ===")

# Load a small sample
pw_kwargs = dict(n_slots=7, max_seq_len=64, max_depth=5, shuffle_facts=True, slot_label_mode="type")
train_loader, val_loader = get_dataloaders(
    dataset_name="proofwriter", batch_size=16, data_dir="nesy_mamba/data",
    max_examples=200, encoding="symbolic", **pw_kwargs,
)

# Check slot label distribution
train_ds = train_loader.dataset
base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
vocab_size = len(base_ds.vocab)

import numpy as np
slot_counts = np.zeros(7)
for i in range(len(train_ds)):
    sl = train_ds[i]["slot_labels"].numpy()
    slot_counts += sl
print(f"\nSlot label distribution (n={len(train_ds)}):")
for k in range(7):
    pct = 100 * slot_counts[k] / len(train_ds)
    print(f"  Slot {k} ({RULE_TYPE_NAMES[k]:12s}): {pct:5.1f}% active")

# Quick model forward pass
n_pos = sum(1 for i in range(len(train_ds)) if train_ds[i]["answer_label"].item() > 0.5)
n_neg = len(train_ds) - n_pos
pos_weight = n_neg / max(n_pos, 1)

cfg = NeSyMambaConfig(
    variant="full", n_layers=2, d_model=64, n_slots=7, max_seq_len=64,
    vocab_size=vocab_size, slot_gate_mode="ema", slot_ema_alpha_init=2.0,
    slot_ema_dynamic_alpha=True, slot_ws_gain=1.0, slot_bias_init=0.0,
    slot_competition=True, slot_temperature=0.5, pos_weight=pos_weight,
)
cfg.lambda_slot = 0.3
cfg.lambda_rule = 0.1
cfg.lambda_ortho = 0.05
cfg.lambda_entropy = 0.5

model = NeSyMamba(cfg)
answer_rules = SymbolicProofWriterDataset.get_answer_rules()

# Forward pass with one batch
batch = next(iter(train_loader))
result = model(
    input_ids=batch["input_ids"],
    answer_labels=batch["answer_label"],
    slot_labels=batch["slot_labels"],
    rules=answer_rules,
    current_epoch=0,
)

print(f"\nForward pass OK:")
print(f"  answer_prob shape: {result['answer_prob'].shape}")
print(f"  slot_values shape: {result['slot_values'].shape}")
print(f"  total_loss: {result['total_loss'].item():.4f}")
print(f"  loss_breakdown: {result['loss_breakdown']}")

# Verify slot_labels are type-based (multiple slots can be active, not just 0..6)
sl = batch["slot_labels"]
print(f"\nBatch slot_labels stats:")
print(f"  shape: {sl.shape}")
print(f"  per-slot mean: {sl.mean(dim=0).tolist()}")
print(f"  any multi-hot: {(sl.sum(dim=1) > 1).any().item()}")

# Backward pass
result["total_loss"].backward()
print(f"\nBackward pass OK - gradients computed")

print("\n=== ALL TESTS PASSED ===")
