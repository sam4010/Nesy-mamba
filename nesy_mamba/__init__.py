"""
Slot-Gated Neuro-Symbolic Mamba (Type-5)
========================================
A self-explaining Mamba model with sigmoid-gated symbolic slots
and differentiable logic losses for neuro-symbolic reasoning.
"""

from .config import NeSyMambaConfig
from .mamba_block import MambaBlock, MambaBackbone
from .slot_gate import SlotGate
from .logic_loss import LogicLossComputer
from .nesy_mamba import NeSyMamba
from .metrics import compute_metrics
from .data_utils import (
    SyntheticRulesDataset,
    ProofWriterDataset,
    SymbolicProofWriterDataset,
    CLUTRRDataset,
    SimpleVocab,
    get_dataloaders,
    load_glove_embeddings,
)

__all__ = [
    "NeSyMambaConfig",
    "MambaBlock",
    "MambaBackbone",
    "SlotGate",
    "LogicLossComputer",
    "NeSyMamba",
    "compute_metrics",
    "SyntheticRulesDataset",
    "ProofWriterDataset",
    "SymbolicProofWriterDataset",
    "CLUTRRDataset",
    "SimpleVocab",
    "get_dataloaders",
]
