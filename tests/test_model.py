"""
Unit tests for Slot-Gated NeSy Mamba.

Covers:
  - Slot gating shapes and firing order
  - Logic loss gradients and properties
  - Mamba block shapes and gradient flow
  - Full model forward/backward pass
  - Loss decrease over gradient steps
  - All 4 ablation variants instantiate correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from nesy_mamba.config import NeSyMambaConfig
from nesy_mamba.mamba_block import MambaBlock, MambaBackbone
from nesy_mamba.slot_gate import SlotGate
from nesy_mamba.logic_loss import LogicLossComputer
from nesy_mamba.nesy_mamba import NeSyMamba
from nesy_mamba.metrics import accuracy, logic_fidelity, slot_fidelity
from nesy_mamba.data_utils import SyntheticRulesDataset, SimpleVocab
from nesy_mamba.probes import extract_features, linear_probe, causal_ablation, compositional_generalisation, HAS_SKLEARN


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return NeSyMambaConfig(
        d_model=32, d_state=8, d_conv=4, n_layers=2,
        expand_factor=2, n_slots=5, vocab_size=50,
        max_seq_len=16, variant="full",
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# ── Mamba Block Tests ───────────────────────────────────────────────

class TestMambaBlock:

    def test_output_shape(self, cfg, device):
        block = MambaBlock(cfg).to(device)
        x = torch.randn(2, 16, cfg.d_model, device=device)
        y = block(x)
        assert y.shape == (2, 16, cfg.d_model)

    def test_gradient_flow(self, cfg, device):
        block = MambaBlock(cfg).to(device)
        x = torch.randn(2, 16, cfg.d_model, device=device, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_backbone_shape(self, cfg, device):
        backbone = MambaBackbone(cfg).to(device)
        x = torch.randn(2, 16, cfg.d_model, device=device)
        y = backbone(x)
        assert y.shape == (2, 16, cfg.d_model)


# ── Slot Gate Tests ─────────────────────────────────────────────────

class TestSlotGate:

    def test_output_shape(self, cfg, device):
        gate = SlotGate(cfg).to(device)
        h = torch.randn(2, 16, cfg.d_model, device=device)
        values, history, firing = gate(h)
        assert values.shape == (2, cfg.n_slots)
        assert history.shape == (2, 16, cfg.n_slots)
        assert firing.shape == (2, cfg.n_slots)

    def test_values_in_range(self, cfg, device):
        gate = SlotGate(cfg).to(device)
        h = torch.randn(4, 16, cfg.d_model, device=device)
        values, _, _ = gate(h)
        assert (values >= 0).all() and (values <= 1).all()

    def test_firing_order_tracked(self, cfg, device):
        gate = SlotGate(cfg).to(device)
        h = torch.randn(1, 16, cfg.d_model, device=device)
        values, _, firing = gate(h)
        # Fired slots should have non-negative timestep
        fired_mask = values > cfg.slot_threshold
        for k in range(cfg.n_slots):
            if fired_mask[0, k]:
                assert firing[0, k].item() >= 0
            else:
                assert firing[0, k].item() == -1

    def test_gradient_flow(self, cfg, device):
        gate = SlotGate(cfg).to(device)
        h = torch.randn(2, 16, cfg.d_model, device=device, requires_grad=True)
        values, _, _ = gate(h)
        values.sum().backward()
        assert h.grad is not None

    def test_explain(self, cfg, device):
        gate = SlotGate(cfg).to(device)
        h = torch.randn(1, 16, cfg.d_model, device=device)
        values, _, firing = gate(h)
        expl = gate.explain(values, firing, [f"R{i}" for i in range(cfg.n_slots)])
        assert isinstance(expl, list)
        assert isinstance(expl[0], list)


# ── Logic Loss Tests ────────────────────────────────────────────────

class TestLogicLoss:

    def test_slot_supervision_gradient(self, cfg, device):
        lc = LogicLossComputer(cfg)
        preds = torch.tensor([[0.8, 0.2, 0.5]], requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        loss = lc.slot_supervision_loss(preds, labels)
        loss.backward()
        assert preds.grad is not None

    def test_slot_supervision_correct(self, device):
        """When preds match labels perfectly → low loss."""
        cfg = NeSyMambaConfig(n_slots=3)
        lc = LogicLossComputer(cfg)
        preds = torch.tensor([[1.0, 0.0, 1.0]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        loss = lc.slot_supervision_loss(preds, labels)
        assert loss.item() < 0.01

    def test_answer_consistency_gradient(self, cfg, device):
        lc = LogicLossComputer(cfg)
        slots = torch.tensor([[0.9, 0.9, 0.1, 0.1, 0.1]], requires_grad=True)
        ans = torch.tensor([0.1], requires_grad=True)
        rules = [(0, 1)]
        loss = lc.answer_consistency_loss(slots, ans, rules)
        loss.backward()
        assert slots.grad is not None
        assert ans.grad is not None

    def test_ortho_loss_identity(self, device):
        """Orthogonality loss should be low when slots are orthogonal."""
        cfg = NeSyMambaConfig(n_slots=3)
        lc = LogicLossComputer(cfg)
        # Orthogonal activations across batch
        preds = torch.eye(3)  # (3, 3) — 3 examples, 3 slots
        loss = lc.orthogonality_loss(preds)
        assert loss.item() < 0.1  # should be near zero

    def test_combined_loss(self, cfg, device):
        lc = LogicLossComputer(cfg).to(device)
        B, K = 4, cfg.n_slots
        preds = torch.rand(B, K, requires_grad=True)
        labels = torch.randint(0, 2, (B, K)).float()
        ans = torch.rand(B)
        total, bd = lc(preds, labels, ans, rules=[(0,), (1, 2)])
        total.backward()
        assert "L_slot" in bd
        assert "L_rule" in bd
        assert "L_ortho" in bd
        assert preds.grad is not None


# ── Full Model Tests ────────────────────────────────────────────────

class TestNeSyMamba:

    def test_forward_pass(self, cfg, device):
        model = NeSyMamba(cfg).to(device)
        B, L = 2, cfg.max_seq_len
        ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
        ans = torch.randint(0, 2, (B,), device=device).float()
        slots = torch.randint(0, 2, (B, cfg.n_slots), device=device).float()

        result = model(ids, ans, slots, rules=[(0,), (1, 2)])
        assert result["answer_prob"].shape == (B,)
        assert result["slot_values"].shape == (B, cfg.n_slots)
        assert result["total_loss"].requires_grad

    def test_backward_pass(self, cfg, device):
        model = NeSyMamba(cfg).to(device)
        B, L = 2, cfg.max_seq_len
        ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
        ans = torch.randint(0, 2, (B,), device=device).float()
        slots = torch.randint(0, 2, (B, cfg.n_slots), device=device).float()

        result = model(ids, ans, slots)
        result["total_loss"].backward()

        # Check all params got gradients
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_loss_decreases(self, cfg, device):
        """Loss should decrease after a few optimisation steps."""
        model = NeSyMamba(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, L = 4, cfg.max_seq_len
        ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
        ans = torch.ones(B, device=device)  # all True
        slots = torch.ones(B, cfg.n_slots, device=device)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            result = model(ids, ans, slots, current_epoch=20)
            result["total_loss"].backward()
            optimizer.step()
            losses.append(result["total_loss"].item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], (
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_explain(self, cfg, device):
        model = NeSyMamba(cfg).to(device)
        ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len), device=device)
        expl = model.explain(ids)
        assert len(expl) == 1
        assert "answer" in expl[0]
        assert "confidence" in expl[0]
        assert "trace" in expl[0]


# ── Ablation Variant Tests ──────────────────────────────────────────

class TestAblationVariants:

    @pytest.mark.parametrize("variant", ["base", "slots_only", "loss_only", "full"])
    def test_variant_instantiates(self, variant, device):
        cfg = NeSyMambaConfig(
            d_model=32, d_state=8, n_layers=2, n_slots=5,
            vocab_size=50, max_seq_len=16, variant=variant,
        )
        model = NeSyMamba(cfg).to(device)
        ids = torch.randint(0, 50, (2, 16), device=device)
        result = model(ids)
        assert result["answer_prob"].shape == (2,)

    def test_base_has_no_slots(self, device):
        cfg = NeSyMambaConfig(
            d_model=32, n_layers=2, variant="base",
            vocab_size=50, max_seq_len=16,
        )
        model = NeSyMamba(cfg)
        assert model.slot_gate is None
        assert model.logic_loss is None

    def test_full_has_all(self, device):
        cfg = NeSyMambaConfig(
            d_model=32, n_layers=2, variant="full",
            vocab_size=50, max_seq_len=16,
        )
        model = NeSyMamba(cfg)
        assert model.slot_gate is not None
        assert model.logic_loss is not None


# ── Metrics Tests ───────────────────────────────────────────────────

class TestMetrics:

    def test_accuracy_perfect(self):
        probs = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        assert accuracy(probs, labels) == 1.0

    def test_logic_fidelity_all_satisfied(self):
        slots = torch.tensor([[0.9, 0.9], [0.1, 0.1]])
        ans = torch.tensor([0.9, 0.9])
        rules = [(0, 1)]
        fid = logic_fidelity(slots, ans, rules)
        assert fid == 1.0

    def test_logic_fidelity_violation(self):
        # Both slots fire but answer is false → violation
        slots = torch.tensor([[0.9, 0.9]])
        ans = torch.tensor([0.1])
        rules = [(0, 1)]
        fid = logic_fidelity(slots, ans, rules)
        assert fid == 0.0

    def test_slot_fidelity_perfect(self):
        preds = torch.tensor([[0.9, 0.1, 0.8]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])
        assert slot_fidelity(preds, labels) == 1.0


# ── Answer Rules Tests ──────────────────────────────────────────────

class TestAnswerRules:

    def test_synthetic_answer_rules(self):
        """get_answer_rules() should return only the conjunction rule."""
        answer_rules = SyntheticRulesDataset.get_answer_rules()
        assert answer_rules == [(0, 1)]

    def test_answer_rules_vs_all_rules(self):
        """answer_rules should be a subset of all rules."""
        all_rules = SyntheticRulesDataset.get_rules()
        answer_rules = SyntheticRulesDataset.get_answer_rules()
        assert len(answer_rules) < len(all_rules)

    def test_logic_fidelity_with_answer_rules(self):
        """Logic fidelity should be high when using correct answer rules."""
        # Example: slot 0 fires, slot 1 doesn't → answer is False
        # With answer_rules [(0,1)], this is vacuously satisfied
        slots = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ans = torch.tensor([0.1])
        answer_rules = [(0, 1)]
        fid = logic_fidelity(slots, ans, answer_rules)
        assert fid == 1.0  # vacuously satisfied

    def test_logic_fidelity_violation_with_answer_rules(self):
        """Logic fidelity should detect violations with answer rules."""
        # Both slots fire but answer is False → violation
        slots = torch.tensor([[0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ans = torch.tensor([0.1])
        answer_rules = [(0, 1)]
        fid = logic_fidelity(slots, ans, answer_rules)
        assert fid == 0.0  # violated: both fire but answer False


# ── SimpleVocab Tests ───────────────────────────────────────────────

class TestSimpleVocab:

    def test_encode_decode(self):
        vocab = SimpleVocab()
        ids = vocab.encode("hello world test", max_len=5)
        assert len(ids) == 5
        assert ids[3] == 0  # padding
        assert ids[4] == 0  # padding

    def test_freeze(self):
        vocab = SimpleVocab()
        vocab.encode("hello world", max_len=4)
        vocab.freeze()
        ids = vocab.encode("unknown word", max_len=4)
        assert ids[0] == SimpleVocab.UNK
        assert ids[1] == SimpleVocab.UNK

    def test_padding(self):
        vocab = SimpleVocab()
        ids = vocab.encode("a", max_len=10)
        assert len(ids) == 10
        assert ids[0] != 0
        assert all(i == 0 for i in ids[1:])


# ── EMA Gate Tests ──────────────────────────────────────────────────

class TestEMAGate:
    """Tests for the EMA slot gate mode (v8)."""

    @pytest.fixture
    def ema_cfg(self):
        """EMA gate with dynamic (input-dependent) alpha — the default."""
        return NeSyMambaConfig(
            d_model=32, d_state=8, d_conv=4, n_layers=2,
            expand_factor=2, n_slots=5, vocab_size=50,
            max_seq_len=16, variant="full",
            slot_gate_mode="ema",
            slot_ema_alpha_init=2.0,
            slot_ema_dynamic_alpha=True,
            slot_ws_gain=1.0,
            slot_bias_init=0.0,
        )

    @pytest.fixture
    def ema_static_cfg(self):
        """EMA gate with static (per-slot constant) alpha — for ablation."""
        return NeSyMambaConfig(
            d_model=32, d_state=8, d_conv=4, n_layers=2,
            expand_factor=2, n_slots=5, vocab_size=50,
            max_seq_len=16, variant="full",
            slot_gate_mode="ema",
            slot_ema_alpha_init=2.0,
            slot_ema_dynamic_alpha=False,
            slot_ws_gain=1.0,
            slot_bias_init=0.0,
        )

    def test_ema_output_shape(self, ema_cfg, device):
        gate = SlotGate(ema_cfg).to(device)
        h = torch.randn(2, 16, ema_cfg.d_model, device=device)
        values, history, firing = gate(h)
        assert values.shape == (2, ema_cfg.n_slots)
        assert history.shape == (2, 16, ema_cfg.n_slots)
        assert firing.shape == (2, ema_cfg.n_slots)

    def test_ema_values_in_range(self, ema_cfg, device):
        gate = SlotGate(ema_cfg).to(device)
        h = torch.randn(4, 16, ema_cfg.d_model, device=device)
        values, _, _ = gate(h)
        assert (values >= 0).all() and (values <= 1).all()

    def test_ema_gradient_flow(self, ema_cfg, device):
        gate = SlotGate(ema_cfg).to(device)
        h = torch.randn(2, 16, ema_cfg.d_model, device=device, requires_grad=True)
        values, _, _ = gate(h)
        values.sum().backward()
        assert h.grad is not None
        # W_alpha should receive gradients (dynamic alpha)
        assert gate.W_alpha.weight.grad is not None

    def test_ema_gradient_flow_static(self, ema_static_cfg, device):
        gate = SlotGate(ema_static_cfg).to(device)
        h = torch.randn(2, 16, ema_static_cfg.d_model, device=device, requires_grad=True)
        values, _, _ = gate(h)
        values.sum().backward()
        assert h.grad is not None
        # Alpha logit should receive gradients (static alpha)
        assert gate.alpha_logit.grad is not None

    def test_ema_slots_vary_with_input(self, ema_cfg, device):
        """EMA slots should produce different values for different inputs
        (unlike monotonic gate which converges to same value)."""
        gate = SlotGate(ema_cfg).to(device)
        # Two very different inputs
        h1 = torch.randn(1, 16, ema_cfg.d_model, device=device) * 3.0
        h2 = -h1  # negated input
        v1, _, _ = gate(h1)
        v2, _, _ = gate(h2)
        # At least some slots should differ between the two inputs
        diff = (v1 - v2).abs().max().item()
        assert diff > 0.01, f"EMA slots should vary with input, but max diff = {diff:.4f}"

    def test_ema_not_monotonic(self, ema_cfg, device):
        """EMA slots should be able to DECREASE (unlike monotonic max gate)."""
        gate = SlotGate(ema_cfg).to(device)
        # First, a strong input to push slots up
        h_strong = torch.ones(1, 8, ema_cfg.d_model, device=device) * 5.0
        # Then, zero input to let slots fall
        h_zero = torch.zeros(1, 8, ema_cfg.d_model, device=device)
        h_combined = torch.cat([h_strong, h_zero], dim=1)  # (1, 16, d)
        _, history, _ = gate(h_combined)
        # Slot values at timestep 8 (after strong input)
        mid_values = history[0, 7, :]  # end of strong input
        end_values = history[0, 15, :]  # end of zero input
        # With EMA, slots should decrease when input goes to zero
        decreased = (end_values < mid_values).any().item()
        assert decreased, "EMA gate should allow slot values to decrease"

    def test_ema_dynamic_alpha_varies_per_token(self, ema_cfg, device):
        """With dynamic alpha, α should vary across timesteps (GRU-like gate)."""
        gate = SlotGate(ema_cfg).to(device)
        h = torch.randn(1, 16, ema_cfg.d_model, device=device)
        # Compute alpha for first and last timestep
        alpha_t0 = torch.sigmoid(gate.W_alpha(h[:, 0, :]))  # (1, K)
        alpha_t15 = torch.sigmoid(gate.W_alpha(h[:, 15, :]))  # (1, K)
        diff = (alpha_t0 - alpha_t15).abs().max().item()
        assert diff > 0.001, f"Dynamic alpha should vary across timesteps, max diff = {diff}"

    def test_ema_static_alpha_constant(self, ema_static_cfg, device):
        """With static alpha, α should be the same regardless of input."""
        gate = SlotGate(ema_static_cfg).to(device)
        assert hasattr(gate, 'alpha_logit'), "Static mode should have alpha_logit parameter"
        assert not hasattr(gate, 'W_alpha'), "Static mode should NOT have W_alpha"

    def test_ema_alpha_staggered(self, ema_cfg, device):
        """Alpha biases should be initialised with staggered values for symmetry breaking."""
        gate = SlotGate(ema_cfg).to(device)
        # Dynamic alpha: check bias staggering
        alphas = torch.sigmoid(gate.W_alpha.bias.data)
        assert alphas.std().item() > 0.01, "Alpha bias values should be staggered"

    def test_ema_full_model_forward(self, ema_cfg, device):
        """Full NeSyMamba with EMA gate should produce valid output."""
        model = NeSyMamba(ema_cfg).to(device)
        ids = torch.randint(1, ema_cfg.vocab_size, (2, 16), device=device)
        labels = torch.tensor([0.0, 1.0], device=device)
        slot_labels = torch.zeros(2, ema_cfg.n_slots, device=device)
        result = model(ids, answer_labels=labels, slot_labels=slot_labels, current_epoch=0)
        assert result["answer_prob"].shape == (2,)
        assert result["slot_values"].shape == (2, ema_cfg.n_slots)
        assert result["total_loss"].item() > 0

    def test_ema_loss_decreases(self, ema_cfg, device):
        """EMA model should learn (loss should decrease)."""
        model = NeSyMamba(ema_cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        ids = torch.randint(1, ema_cfg.vocab_size, (8, 16), device=device)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], device=device)
        slot_labels = torch.zeros(8, ema_cfg.n_slots, device=device)

        model.train()
        first_loss = None
        for step in range(20):
            opt.zero_grad()
            result = model(ids, answer_labels=labels, slot_labels=slot_labels, current_epoch=step)
            loss = result["total_loss"]
            if step == 0:
                first_loss = loss.item()
            loss.backward()
            opt.step()
        assert loss.item() < first_loss, f"Loss should decrease: {first_loss:.4f} -> {loss.item():.4f}"


# ── Config Tests ────────────────────────────────────────────────────

class TestConfig:

    def test_to_dict(self):
        cfg = NeSyMambaConfig(d_model=32, n_layers=2)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["d_model"] == 32
        assert d["n_layers"] == 2

    def test_from_dict(self):
        d = {"d_model": 64, "n_layers": 4, "unknown_key": True}
        cfg = NeSyMambaConfig.from_dict(d)
        assert cfg.d_model == 64
        assert cfg.n_layers == 4

    def test_roundtrip(self):
        cfg1 = NeSyMambaConfig(d_model=128, variant="slots_only")
        cfg2 = NeSyMambaConfig.from_dict(cfg1.to_dict())
        assert cfg1.d_model == cfg2.d_model
        assert cfg1.variant == cfg2.variant

    def test_ema_config_fields(self):
        """New v8 config fields should serialize correctly."""
        cfg = NeSyMambaConfig(
            slot_gate_mode="ema",
            slot_ema_alpha_init=1.5,
            slot_ema_dynamic_alpha=True,
            slot_ws_gain=0.5,
        )
        d = cfg.to_dict()
        assert d["slot_gate_mode"] == "ema"
        assert d["slot_ema_alpha_init"] == 1.5
        assert d["slot_ema_dynamic_alpha"] is True
        assert d["slot_ws_gain"] == 0.5
        cfg2 = NeSyMambaConfig.from_dict(d)
        assert cfg2.slot_gate_mode == "ema"
        assert cfg2.slot_ema_dynamic_alpha is True


# ── Probes Tests ────────────────────────────────────────────────────

class TestProbes:
    """Tests for interpretability probes (feature extraction, linear probe, ablation)."""

    @pytest.fixture
    def trained_model_and_loader(self):
        """Create a small model, train for a few steps, return model + loader."""
        cfg = NeSyMambaConfig(
            variant="full", d_model=32, n_slots=5, n_layers=1,
            vocab_size=50, max_seq_len=16, lr=0.01,
        )
        model = NeSyMamba(cfg)
        ds = SyntheticRulesDataset(n_examples=64, vocab_size=50, seq_len=16, n_slots=5)
        loader = torch.utils.data.DataLoader(ds, batch_size=16)
        rules = SyntheticRulesDataset.get_rules()

        # Quick train so features aren't random
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(5):
            for batch in loader:
                optimizer.zero_grad()
                result = model(
                    input_ids=batch["input_ids"],
                    answer_labels=batch["answer_label"],
                    slot_labels=batch["slot_labels"],
                    rules=rules,
                    current_epoch=10,
                )
                result["total_loss"].backward()
                optimizer.step()

        return model, loader, cfg

    def test_extract_features_shapes(self, trained_model_and_loader):
        model, loader, cfg = trained_model_and_loader
        features = extract_features(model, loader, torch.device("cpu"), max_examples=32)

        assert features["slot_values"].shape[0] <= 32
        assert features["slot_values"].shape[1] == cfg.n_slots
        assert features["h_final"].shape[1] == cfg.d_model
        assert features["answer_labels"].shape[0] == features["slot_values"].shape[0]
        assert features["slot_labels"].shape == features["slot_values"].shape

    def test_extract_features_no_limit(self, trained_model_and_loader):
        model, loader, cfg = trained_model_and_loader
        features = extract_features(model, loader, torch.device("cpu"), max_examples=0)
        assert features["slot_values"].shape[0] == 64  # full dataset

    @pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
    def test_linear_probe_runs(self, trained_model_and_loader):
        model, loader, cfg = trained_model_and_loader
        features = extract_features(model, loader, torch.device("cpu"), max_examples=0)
        results = linear_probe(features, n_cv_folds=3)

        assert "per_rule" in results
        assert len(results["per_rule"]) == cfg.n_slots
        assert "mean_slot_acc" in results
        assert "mean_hidden_acc" in results
        assert "slot_wins" in results
        assert 0.0 <= results["mean_slot_acc"] <= 1.0
        assert 0.0 <= results["mean_hidden_acc"] <= 1.0

    def test_causal_ablation_runs(self, trained_model_and_loader):
        model, loader, cfg = trained_model_and_loader
        results = causal_ablation(model, loader, torch.device("cpu"), max_examples=32)

        assert "baseline_acc" in results
        assert "zero_all_acc" in results
        assert "shuffle_acc" in results
        assert len(results["per_slot_zero"]) == cfg.n_slots
        assert 0.0 <= results["baseline_acc"] <= 1.0
        assert 0.0 <= results["zero_all_acc"] <= 1.0

    def test_causal_ablation_base_variant(self):
        """Base variant (no slots) should return error dict."""
        cfg = NeSyMambaConfig(variant="base", d_model=32, n_slots=5, n_layers=1,
                               vocab_size=50, max_seq_len=16)
        model = NeSyMamba(cfg)
        ds = SyntheticRulesDataset(n_examples=16, vocab_size=50, seq_len=16, n_slots=5)
        loader = torch.utils.data.DataLoader(ds, batch_size=8)
        results = causal_ablation(model, loader, torch.device("cpu"))
        assert "error" in results

    def test_compositional_generalisation_runs(self, trained_model_and_loader):
        model, loader, cfg = trained_model_and_loader
        features = extract_features(model, loader, torch.device("cpu"))
        results = compositional_generalisation(model, features, torch.device("cpu"))

        assert "overall_acc" in results
        assert "per_depth" in results
        assert 0.0 <= results["overall_acc"] <= 1.0

    def test_attribution_no_grad_leak(self):
        """Verify slot_token_attribution doesn't leak gradients to model params."""
        cfg = NeSyMambaConfig(variant="full", d_model=32, n_slots=4, n_layers=1,
                               vocab_size=20, max_seq_len=16)
        model = NeSyMamba(cfg)
        x = torch.randint(1, 20, (2, 16))
        attr = model.slot_token_attribution(x, slot_idx=0, n_steps=3)

        assert attr.shape == (2, 16)
        # No param should have dirty gradients
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() == 0, f"{name} has dirty gradients after attribution"
        # All params (except buffers) should still require grad
        for name, p in model.named_parameters():
            if "_pos_weight" not in name:
                assert p.requires_grad, f"{name} lost requires_grad after attribution"

    def test_loss_only_variant_has_logic_loss(self):
        """Verify loss_only variant produces logic loss terms (not just pass)."""
        cfg = NeSyMambaConfig(variant="loss_only", d_model=32, n_slots=4,
                               n_layers=1, vocab_size=20, max_seq_len=16)
        model = NeSyMamba(cfg)
        x = torch.randint(1, 20, (4, 16))
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        rules = [(0, 1)]
        out = model(input_ids=x, answer_labels=labels, rules=rules, current_epoch=10)

        # Should have logic loss components beyond just L_task
        logic_keys = [k for k in out["loss_breakdown"] if k.startswith("L_") and k not in ("L_task", "L_total")]
        assert len(logic_keys) > 0, f"loss_only should produce logic losses, got: {out['loss_breakdown'].keys()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
