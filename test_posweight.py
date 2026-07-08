"""Quick test: verify per-slot pos_weight flows through correctly."""
import torch, sys
sys.path.insert(0, ".")
from nesy_mamba.config import NeSyMambaConfig
from nesy_mamba.nesy_mamba import NeSyMamba
from nesy_mamba.data_utils import get_dataloaders, SymbolicProofWriterDataset, RULE_TYPE_NAMES
import numpy as np

# Load small sample
pw_kwargs = dict(n_slots=7, max_seq_len=64, max_depth=5, shuffle_facts=True, slot_label_mode="type")
train_loader, val_loader = get_dataloaders(
    dataset_name="proofwriter", batch_size=16, data_dir="nesy_mamba/data",
    max_examples=500, encoding="symbolic", **pw_kwargs,
)
train_ds = train_loader.dataset
base_ds = train_ds.dataset if hasattr(train_ds, 'dataset') else train_ds
vocab_size = len(base_ds.vocab)

# Compute slot label freq
slot_counts = np.zeros(7)
n_sample = len(train_ds)
for i in range(n_sample):
    sl = train_ds[i]["slot_labels"].numpy()
    slot_counts += sl
slot_label_freq = slot_counts / n_sample
print("Slot label frequencies:")
for k in range(7):
    print(f"  {RULE_TYPE_NAMES[k]:12s}: {slot_label_freq[k]:.3f}")

# Build model
cfg = NeSyMambaConfig(
    variant="full", n_layers=2, d_model=64, n_slots=7, max_seq_len=64,
    vocab_size=vocab_size, slot_gate_mode="ema", slot_ema_alpha_init=2.0,
    slot_ema_dynamic_alpha=True, slot_ws_gain=1.0, slot_bias_init=0.0,
    slot_competition=True, slot_temperature=0.5, pos_weight=0.8367,
)
cfg.lambda_slot = 1.0
cfg.lambda_rule = 0.1
cfg.lambda_ortho = 0.05
cfg.lambda_entropy = 0.5

model = NeSyMamba(cfg)

# Set pos_weight
freq_tensor = torch.tensor(slot_label_freq, dtype=torch.float32)
model.set_slot_pos_weight(freq_tensor)
pw = model.logic_loss.slot_pos_weight
print("\nPer-slot pos_weight:")
for k in range(7):
    print(f"  {RULE_TYPE_NAMES[k]:12s}: freq={slot_label_freq[k]:.3f} -> w={pw[k]:.1f}")

# Forward pass
batch = next(iter(train_loader))
answer_rules = SymbolicProofWriterDataset.get_answer_rules()
result = model(
    input_ids=batch["input_ids"],
    answer_labels=batch["answer_label"],
    slot_labels=batch["slot_labels"],
    rules=answer_rules,
    current_epoch=0,
)
print(f"\nForward OK: L_total={result['total_loss'].item():.4f}")
print(f"  L_slot={result['loss_breakdown'].get('L_slot', 0):.4f}")
print(f"  slot_values mean={result['slot_values'].mean(dim=0).tolist()}")

# Backward
result['total_loss'].backward()
print("Backward OK")

# Verify gradients exist for slot gate params
for name, p in model.named_parameters():
    if 'slot' in name and p.grad is not None:
        grad_norm = p.grad.norm().item()
        if grad_norm > 0.01:
            print(f"  {name}: grad_norm={grad_norm:.4f}")

print("\nALL TESTS PASSED")
