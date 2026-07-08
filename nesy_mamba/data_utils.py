"""
Data Utilities for NeSy Mamba.

Includes:
  - Synthetic rule-based dataset for smoke-testing
  - ProofWriter dataset loader (text and symbolic encodings)
  - CLUTRR dataset loader
  - Simple vocabulary builder
  - Rule type classifier for supervised slot labels
"""

import random
import json
import csv
import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ── Rule Type Classifier ───────────────────────────────────────────
#
# Classifies ProofWriter rules into 7 semantic types based on their
# logical structure (antecedent/consequent pattern).  This provides
# *globally consistent* slot assignments across all theories, so
# slot_k always means the same rule archetype.
#
# Taxonomy (derived from depth-5 CWA corpus analysis):
#   Slot 0: prop(1)->prop        "Property Implication"   (33.4%)
#   Slot 1: prop+prop(2)->prop   "Property Conjunction"   (27.0%)
#   Slot 2: rel(1)->rel          "Relation Chain"         (10.9%)
#   Slot 3: prop(1)->rel         "Property-to-Relation"   ( 5.8%)
#   Slot 4: rel(1)->prop         "Relation-to-Property"   ( 5.5%)
#   Slot 5: *(2)->rel            "Mixed/Rel→Relation"     (10.9%)
#   Slot 6: *(2)->prop (+other)  "Mixed/Rel→Property"     ( 6.5%)
# ────────────────────────────────────────────────────────────────────

_CLAUSE_RE = re.compile(r'\("([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\)')

RULE_TYPE_NAMES = [
    "PropImpl",       # prop(1)->prop
    "PropConj",       # prop+prop(2)->prop
    "RelChain",       # rel(1)->rel
    "Prop2Rel",       # prop(1)->rel
    "Rel2Prop",       # rel(1)->prop
    "Mixed2Rel",      # *(2)->rel
    "Mixed2Prop",     # *(2)->prop / other
]

NUM_RULE_TYPES = len(RULE_TYPE_NAMES)  # 7


def classify_rule(representation: str) -> int:
    """Classify a ProofWriter rule representation into one of 7 types.

    Args:
        representation: Rule representation string, e.g.
            '((("something" "is" "big" "+")) -> ("something" "is" "kind" "+"))'

    Returns:
        int: Slot/type index in [0, 6].
    """
    if "->" not in representation:
        return 6  # fallback

    ante_str, cons_str = representation.split("->", 1)
    ante_clauses = _CLAUSE_RE.findall(ante_str)
    cons_clauses = _CLAUSE_RE.findall(cons_str)

    n_ante = len(ante_clauses)

    # Classify antecedent clause types
    ante_has_prop = any(rel.lower().strip() == "is" for _, rel, _, _ in ante_clauses)
    ante_has_rel = any(rel.lower().strip() != "is" for _, rel, _, _ in ante_clauses)

    # Classify consequent type
    cons_is_prop = all(rel.lower().strip() == "is" for _, rel, _, _ in cons_clauses) if cons_clauses else True
    cons_is_rel = not cons_is_prop

    # Single-antecedent rules (n_ante == 1)
    if n_ante == 1:
        if ante_has_prop and not ante_has_rel:
            # prop(1) -> ?
            return 0 if cons_is_prop else 3  # PropImpl or Prop2Rel
        else:
            # rel(1) -> ?
            return 2 if cons_is_rel else 4   # RelChain or Rel2Prop

    # Multi-antecedent rules (n_ante >= 2)
    if ante_has_prop and not ante_has_rel:
        # prop+prop(2) -> prop
        return 1 if cons_is_prop else 5  # PropConj or (rare) prop+prop->rel→Mixed2Rel
    else:
        # Mixed or rel+rel antecedents
        return 5 if cons_is_rel else 6  # Mixed2Rel or Mixed2Prop


def classify_theory_rules(rules_dict: dict) -> dict:
    """Classify all rules in a theory, returning {rule_key: type_idx}.

    Args:
        rules_dict: ProofWriter theory["rules"] dict.

    Returns:
        dict mapping rule key (e.g. "rule1") to type index (0-6).
    """
    mapping = {}
    for rkey, rval in rules_dict.items():
        rep = rval.get("representation", "")
        mapping[rkey] = classify_rule(rep)
    return mapping


# ── Synthetic Rule Dataset ──────────────────────────────────────────

class SyntheticRulesDataset(Dataset):
    """
    Synthetic logical reasoning dataset for smoke-testing.

    Generates sequences of token IDs representing "facts" with
    deterministic rules. Each example has:
      - input_ids:    (seq_len,) token sequence
      - answer_label: 0 or 1 (True/False)
      - slot_labels:  (n_slots,) which rules fired

    Rules (hard-coded for 5 rules + 2 buffer):
      Rule 0: token 10 present -> slot 0 fires
      Rule 1: token 20 present -> slot 1 fires
      Rule 2: tokens 10 AND 20 both present -> slot 2 fires
      Rule 3: token 30 present -> slot 3 fires
      Rule 4: token 40 present -> slot 4 fires
      Answer is True if slot 0 AND slot 1 both fire (conjunction rule).
    """

    def __init__(
        self,
        n_examples: int = 1000,
        seq_len: int = 32,
        vocab_size: int = 50,
        n_slots: int = 7,
        seed: int = 42,
    ):
        super().__init__()
        self.n_examples = n_examples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_slots = n_slots

        rng = random.Random(seed)
        self.data = []

        trigger_tokens = {0: 10, 1: 20, 2: None, 3: 30, 4: 40}
        trigger_set = {v for v in trigger_tokens.values() if v is not None}
        safe_tokens = [t for t in range(1, vocab_size) if t not in trigger_set]

        for _ in range(n_examples):
            # Random base sequence -- EXCLUDE trigger tokens to avoid label noise
            tokens = [rng.choice(safe_tokens) for _ in range(seq_len)]

            # Randomly inject trigger tokens
            slot_labels = [0.0] * n_slots
            for slot_id, trigger in trigger_tokens.items():
                if trigger is not None and rng.random() > 0.5:
                    pos = rng.randint(0, seq_len - 1)
                    tokens[pos] = trigger
                    slot_labels[slot_id] = 1.0

            # Rule 2: conjunction of Rule 0 and Rule 1
            if slot_labels[0] == 1.0 and slot_labels[1] == 1.0:
                slot_labels[2] = 1.0

            # Answer: True if Rule 0 AND Rule 1 fire
            answer = 1.0 if (slot_labels[0] == 1.0 and slot_labels[1] == 1.0) else 0.0

            # Buffer slots (5, 6) stay 0
            self.data.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "answer_label": torch.tensor(answer, dtype=torch.float32),
                "slot_labels": torch.tensor(slot_labels, dtype=torch.float32),
            })

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_rules() -> list:
        """Return rule definitions (antecedent slot indices)."""
        return [
            (0,),      # Rule 0: single antecedent
            (1,),      # Rule 1: single antecedent
            (0, 1),    # Rule 2: conjunction
            (3,),      # Rule 3: single antecedent
            (4,),      # Rule 4: single antecedent
        ]

    @staticmethod
    def get_rule_names() -> list:
        """Human-readable rule names."""
        return [
            "HasToken10",
            "HasToken20",
            "HasToken10_AND_Token20",
            "HasToken30",
            "HasToken40",
            "Buffer_0",
            "Buffer_1",
        ]

    @staticmethod
    def get_answer_rules() -> list:
        """Rules that logically imply the answer (for L_rule and logic fidelity).

        In this synthetic dataset, the answer is True iff Rule 0 AND Rule 1
        both fire (i.e., both token 10 and token 20 are present).
        Single-slot rules do NOT individually imply the answer.
        """
        return [(0, 1)]


# ── Simple Vocabulary ───────────────────────────────────────────────

class SimpleVocab:
    """Simple word-level vocabulary for text tokenization."""

    PAD = 0
    UNK = 1

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self._frozen = False

    # Regex pattern: splits on whitespace but separates punctuation as tokens
    _TOKENIZE_RE = re.compile(r"""
        \w+        # word characters (letters, digits, underscore)
        | [.,;:!?] # common punctuation as separate tokens
        | [-'"]    # hyphens, quotes
    """, re.VERBOSE)

    def add(self, word: str) -> int:
        if word not in self.word2idx:
            if self._frozen:
                return self.UNK
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        return self.word2idx[word]

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Tokenize text with punctuation awareness.

        Splits 'Bob is nice.' → ['bob', 'is', 'nice', '.']
        Keeps entity names intact, separates punctuation marks.
        """
        return cls._TOKENIZE_RE.findall(text.lower())

    def encode(self, text: str, max_len: int) -> list:
        """Tokenize text into padded ID list."""
        tokens = self.tokenize(text)
        ids = [self.add(w) for w in tokens[:max_len]]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def freeze(self):
        """Freeze vocabulary -- new words map to UNK."""
        self._frozen = True

    def __len__(self) -> int:
        return len(self.word2idx)

    def size(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path: str) -> "SimpleVocab":
        v = cls()
        with open(path) as f:
            v.word2idx = json.load(f)
        v.idx2word = {i: w for w, i in v.word2idx.items()}
        v._frozen = True
        return v


def load_glove_embeddings(
    glove_path: str,
    vocab: "SimpleVocab",
    embed_dim: int,
) -> torch.Tensor:
    """
    Load GloVe vectors and build an embedding matrix aligned with vocab.

    Args:
        glove_path: path to glove.6B.100d.txt (or any GloVe file)
        vocab: our SimpleVocab (word2idx mapping)
        embed_dim: model's d_model dimension

    Returns:
        weight: (vocab_size, embed_dim) tensor
    """
    glove_vectors = {}
    target_words = {w.lower() for w in vocab.word2idx}

    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in target_words:
                glove_vectors[word] = [float(x) for x in parts[1:]]

    glove_dim = len(next(iter(glove_vectors.values()))) if glove_vectors else 100
    vocab_size = len(vocab)

    weight = torch.empty(vocab_size, embed_dim)
    nn.init.xavier_uniform_(weight)

    found = 0
    for word, idx in vocab.word2idx.items():
        key = word.lower()
        if key in glove_vectors:
            vec = torch.tensor(glove_vectors[key], dtype=torch.float32)
            if glove_dim >= embed_dim:
                weight[idx] = vec[:embed_dim]
            else:
                weight[idx, :glove_dim] = vec
            found += 1

    weight[SimpleVocab.PAD] = 0.0

    print(f"  GloVe: {found}/{vocab_size} words matched "
          f"(dim {glove_dim}->{embed_dim})")
    return weight


# ── ProofWriter Dataset (text encoding) ─────────────────────────────

class ProofWriterDataset(Dataset):
    """
    ProofWriter dataset loader (CWA format) -- raw natural-language text.

    Each JSONL line is a *theory* containing:
      - "theory": natural-language context (facts + rules)
      - "questions": dict of Q1..QN, each with "question", "answer", "QDep", "proofs"
      - "rules": dict of rule1..ruleN with "text" and "representation"

    We flatten each theory into individual (context+question, answer, slot_labels) examples.
    """

    SPLIT_MAP = {
        "train": "meta-train.jsonl",
        "val": "meta-dev.jsonl",
        "dev": "meta-dev.jsonl",
        "test": "meta-test.jsonl",
    }

    SUBDIRS = [
        "CWA/depth-3ext-NatLang",
        "CWA/depth-5",
        "CWA/depth-3ext",
        "CWA/depth-3",
        "CWA/depth-2",
        "CWA/depth-1",
        "CWA/depth-0",
        "OWA/depth-5",
        "",
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_depth: int = 5,
        max_seq_len: int = 512,
        n_slots: int = 7,
        vocab=None,
        subset=None,
        slot_label_mode: str = "type",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_slots = n_slots
        self.vocab = vocab or SimpleVocab()
        self.rule_templates = []
        self.slot_label_mode = slot_label_mode

        filename = self.SPLIT_MAP.get(split, f"meta-{split}.jsonl")

        if subset:
            search_dirs = [subset]
        else:
            search_dirs = self.SUBDIRS

        data_path = None
        for subdir in search_dirs:
            candidate = os.path.join(data_dir, subdir, filename)
            if os.path.exists(candidate):
                data_path = candidate
                break

        if data_path is None:
            searched = [os.path.join(data_dir, s, filename) for s in search_dirs]
            raise FileNotFoundError(
                f"ProofWriter data not found for split='{split}'. Searched:\n"
                + "\n".join(f"  - {c}" for c in searched)
                + f"\n\nDownload CWA data from: https://allenai.org/data/proofwriter"
                + f"\nExtract into: {data_dir}/"
            )

        self.data = []
        n_theories = 0
        n_skipped = 0

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                theory = json.loads(line)
                n_theories += 1

                context = theory.get("theory", "")
                questions = theory.get("questions", {})
                rules = theory.get("rules", {})
                n_rules = len(rules)

                # Build rule type mapping for this theory
                rule_type_map = classify_theory_rules(rules) if self.slot_label_mode == "type" else {}

                rule_texts = []
                for rkey in sorted(rules.keys(),
                                   key=lambda k: int(re.search(r'\d+', k).group())
                                   if re.search(r'\d+', k) else 0):
                    rule_texts.append(rules[rkey].get("text", ""))

                for qkey, qval in questions.items():
                    depth = qval.get("QDep", 0)
                    if isinstance(depth, str):
                        try:
                            depth = int(depth)
                        except ValueError:
                            depth = 0

                    if depth > max_depth:
                        n_skipped += 1
                        continue

                    question_text = qval.get("question", "")
                    answer = qval.get("answer", False)
                    proofs = qval.get("proofs", "")

                    text = context + " [SEP] " + question_text
                    input_ids = self.vocab.encode(text, max_seq_len)

                    if isinstance(answer, bool):
                        answer_val = 1.0 if answer else 0.0
                    elif isinstance(answer, str):
                        answer_val = 1.0 if answer.lower() == "true" else 0.0
                    else:
                        answer_val = float(answer)

                    slot_labels = self._extract_slot_labels(
                        proofs, n_rules, rule_type_map=rule_type_map
                    )

                    self.data.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "answer_label": torch.tensor(answer_val, dtype=torch.float32),
                        "slot_labels": torch.tensor(slot_labels, dtype=torch.float32),
                    })

        print(f"  ProofWriter [{split}]: {len(self.data)} examples "
              f"from {n_theories} theories (skipped {n_skipped} deep)")

    def _extract_slot_labels(self, proofs, n_rules: int, rule_type_map: dict = None) -> list:
        """Extract which rule *types* fired from the proof string.

        In "type" mode (default), rules referenced in the proof are mapped
        to their semantic type (0-6) via ``rule_type_map``.  This gives
        globally consistent slot labels: slot_k always means the same
        rule archetype across all theories.

        In "index" mode (legacy), rules are mapped positionally:
        rule_i → slot_i (as before).
        """
        labels = [0.0] * self.n_slots
        proof_str = str(proofs) if proofs else ""
        for m in re.finditer(r"rule(\d+)", proof_str, re.IGNORECASE):
            rule_key = f"rule{m.group(1)}"
            if self.slot_label_mode == "type" and rule_type_map:
                type_idx = rule_type_map.get(rule_key, NUM_RULE_TYPES - 1)
                if 0 <= type_idx < self.n_slots:
                    labels[type_idx] = 1.0
            else:
                # Legacy index-based mapping
                idx = int(m.group(1)) - 1
                if 0 <= idx < self.n_slots:
                    labels[idx] = 1.0
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_answer_rules(n_slots: int = NUM_RULE_TYPES) -> list:
        """Each rule type can independently contribute to the answer.

        With type-based slots, we have K rule type slots.  Each can
        independently fire without requiring conjuncts from other types.
        Only returns rules for slots that actually exist (indices < n_slots).
        """
        return [(i,) for i in range(min(n_slots, NUM_RULE_TYPES))]

    @staticmethod
    def get_rule_names() -> list:
        return list(RULE_TYPE_NAMES)


# ── Symbolic Encoding Helpers ───────────────────────────────────────

# Verb-stem normalization so triple representations ("needs") match
# question text ("need" after "does not").
_VERB_STEMS = {
    "needs": "need", "need": "need",
    "likes": "like", "like": "like",
    "sees": "see", "see": "see",
    "chases": "chase", "chase": "chase",
    "visits": "visit", "visit": "visit",
    "eats": "eat", "eat": "eat",
}

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were",
    "they", "them", "their", "then", "that", "it",
    "if", "and", "or", "someone", "something",
})


def _normalize_token(word):
    """Lowercase, strip punctuation, stem known verbs."""
    w = word.lower().strip('.,!?;:()"')
    return _VERB_STEMS.get(w, w)


def _parse_triple_repr(repr_str):
    """Parse a ProofWriter triple representation into normalised tokens.

    '("Bob" "is" "round" "+")' -> ['bob', 'round']
    '("The dog" "needs" "the mouse" "+")' -> ['dog', 'need', 'mouse']
    """
    parts = re.findall(r'"([^"]*)"', repr_str)
    if len(parts) < 3:
        return []
    entity, relation, obj = parts[0], parts[1], parts[2]

    tokens = []
    for w in entity.split():
        nw = _normalize_token(w)
        if nw and nw not in _STOP_WORDS:
            tokens.append(nw)

    rel = _normalize_token(relation)
    if rel not in _STOP_WORDS:
        tokens.append(rel)

    for w in obj.split():
        nw = _normalize_token(w)
        if nw and nw not in _STOP_WORDS:
            tokens.append(nw)
    return tokens


def _parse_rule_repr(repr_str, entity_map=None):
    """Parse a ProofWriter rule representation.

    Returns (antecedent_tokens, consequent_tokens).
    '((("someone" "is" "big" "+") ...) -> ("someone" "is" "kind" "+"))'
    -> (['big', ...], ['kind'])

    When *entity_map* is provided, entity names appearing in the object
    position of non-"is" clauses are replaced with their anonymous tokens.
    """
    if "->" not in repr_str:
        return [], []
    ante_str, cons_str = repr_str.split("->", 1)

    def _extract(s):
        clauses = re.findall(
            r'\("([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"\)', s
        )
        toks = []
        for _entity, relation, obj, _pol in clauses:
            rel = _normalize_token(relation)
            if rel not in _STOP_WORDS:
                toks.append(rel)

            # If relation is NOT "is", the obj is likely an entity.
            obj_norm = _normalize_entity(obj)
            if entity_map and obj_norm in entity_map:
                toks.append(entity_map[obj_norm])
            else:
                for w in obj.split():
                    nw = _normalize_token(w)
                    if nw and nw not in _STOP_WORDS:
                        toks.append(nw)
        return toks

    return _extract(ante_str), _extract(cons_str)


def _parse_question_text(text):
    """Parse question text into normalised tokens + negation flag.

    'Bob is nice.'                      -> (['bob', 'nice'], False)
    'Charlie is not red.'               -> (['charlie', 'red'], True)
    'The dog does not need the mouse.'  -> (['dog', 'need', 'mouse'], True)
    """
    text = text.lower().strip().rstrip(".")
    words = text.split()

    negated = False
    clean = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("the", "a", "an"):
            i += 1
            continue
        if w in ("does", "do") and i + 1 < len(words) and words[i + 1] == "not":
            negated = True
            i += 2
            continue
        if w in ("is", "are") and i + 1 < len(words) and words[i + 1] == "not":
            negated = True
            i += 2
            continue
        if w in ("is", "are"):
            i += 1
            continue
        clean.append(_normalize_token(w))
        i += 1

    return clean, negated


# ── Entity Normalisation ────────────────────────────────────────────

def _normalize_entity(name):
    """Normalise an entity name: lowercase, strip articles.

    'Bob'            -> 'bob'
    'The dog'        -> 'dog'
    'The bald eagle' -> 'bald eagle'
    """
    words = name.lower().split()
    clean = [w for w in words if w not in ("the", "a", "an")]
    return " ".join(clean)


def _extract_question_entities(text):
    """Extract entity candidates from a ProofWriter question string.

    Returns a list of entity names (normalised, lowercase, no articles).

    Patterns handled:
        'Bob is nice.'                     -> ['bob']
        'Charlie is not red.'              -> ['charlie']
        'The dog does not need the mouse.' -> ['dog', 'mouse']
        'The bald eagle sees the lion.'    -> ['bald eagle', 'lion']
    """
    text = text.lower().strip().rstrip(".")
    words = text.split()

    # --- subject: everything before the first copula / auxiliary -----
    subject_words = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("is", "are", "does", "do"):
            break
        if w not in ("the", "a", "an"):
            subject_words.append(w)
        i += 1

    entities = []
    if subject_words:
        entities.append(" ".join(subject_words))

    # --- object: only for relational (does/do) sentences -------------
    if i < len(words) and words[i] in ("does", "do"):
        j = i + 1
        if j < len(words) and words[j] == "not":
            j += 1
        if j < len(words):
            j += 1  # skip the main verb (need, see, chase …)
        object_words = []
        while j < len(words):
            w = words[j]
            if w not in ("the", "a", "an"):
                object_words.append(w)
            j += 1
        if object_words:
            entities.append(" ".join(object_words))

    return entities


# ── Symbolic ProofWriter Dataset ────────────────────────────────────

_MAX_ENTITIES = 15

class SymbolicProofWriterDataset(Dataset):
    """
    ProofWriter with structured symbolic encoding + entity anonymisation.

    Each theory's named entities (Bob, Charlie, ...) are replaced with
    positional identifiers ``[E1]``, ``[E2]``, ... so the model cannot
    memorise entity-specific patterns and must learn structural reasoning.

    Encoding example::

        [FACT] [E1] round [FACT] [E1] nice [FACT] [E2] red
        [RULE] big blue young [IMP] kind
        [SEP] [E1] nice

    Property triples  ("Bob" "is" "round" "+") become ``[FACT] [Ei] property``.
    Relational triples ("The dog" "needs" "the mouse" "+") become
    ``[FACT] [Ei] verb [Ej]``.
    Rules keep only property/relation tokens (no entity names).
    """

    SPLIT_MAP = ProofWriterDataset.SPLIT_MAP
    SUBDIRS = ProofWriterDataset.SUBDIRS

    SPECIAL = (
        ["[FACT]", "[RULE]", "[IMP]", "[SEP]"]
        + [f"[E{i}]" for i in range(1, _MAX_ENTITIES + 1)]
    )

    def __init__(
        self,
        data_dir,
        split="train",
        max_depth=5,
        max_seq_len=64,
        n_slots=7,
        vocab=None,
        subset=None,
        shuffle_facts=False,
        seed=42,
        slot_label_mode="type",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_slots = n_slots
        self.vocab = vocab or SimpleVocab()
        self._rng = random.Random(seed) if shuffle_facts else None
        self.slot_label_mode = slot_label_mode

        for tok in self.SPECIAL:
            self.vocab.add(tok)

        filename = self.SPLIT_MAP.get(split, f"meta-{split}.jsonl")

        search_dirs = [subset] if subset else self.SUBDIRS
        data_path = None
        for subdir in search_dirs:
            candidate = os.path.join(data_dir, subdir, filename)
            if os.path.exists(candidate):
                data_path = candidate
                break

        if data_path is None:
            searched = [os.path.join(data_dir, s, filename) for s in search_dirs]
            raise FileNotFoundError(
                f"ProofWriter data not found for split='{split}'. Searched:\n"
                + "\n".join(f"  - {c}" for c in searched)
            )

        self.data = []
        n_theories = 0
        n_skipped = 0
        total_toks = 0
        n_true = 0
        n_false = 0
        from collections import Counter
        depth_counts = Counter()

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                theory = json.loads(line)
                n_theories += 1

                triples = theory.get("triples", {})
                rules = theory.get("rules", {})
                questions = theory.get("questions", {})
                n_rules = len(rules)

                # Build rule type mapping for this theory
                rule_type_map = classify_theory_rules(rules) if self.slot_label_mode == "type" else {}

                # Collect entities and build anonymous mapping
                entity_map = self._build_entity_map(
                    triples, rules, questions,
                )

                # Build theory prefix with anonymous entities
                prefix = self._build_theory_tokens(
                    triples, rules, entity_map, self._rng,
                )

                for qkey, qval in questions.items():
                    depth = qval.get("QDep", 0)
                    if isinstance(depth, str):
                        try:
                            depth = int(depth)
                        except ValueError:
                            depth = 0
                    if depth > max_depth:
                        n_skipped += 1
                        continue

                    question_text = qval.get("question", "")
                    answer = qval.get("answer", False)
                    proofs = qval.get("proofs", "")

                    q_tokens, negated = self._anon_question(
                        question_text, entity_map,
                    )
                    # Query FIRST: causal SSM needs to know what to look
                    # for before scanning the facts.
                    full_tokens = q_tokens + ["[SEP]"] + prefix

                    ids = [self.vocab.add(t) for t in full_tokens[:max_seq_len]]
                    total_toks += len(ids)
                    ids += [SimpleVocab.PAD] * (max_seq_len - len(ids))

                    if isinstance(answer, bool):
                        answer_val = 1.0 if answer else 0.0
                    elif isinstance(answer, str):
                        answer_val = 1.0 if answer.lower() == "true" else 0.0
                    else:
                        answer_val = float(answer)
                    if negated:
                        answer_val = 1.0 - answer_val

                    slot_labels = self._extract_slot_labels(
                        proofs, n_rules, rule_type_map=rule_type_map
                    )

                    depth_counts[depth] += 1
                    if answer_val > 0.5:
                        n_true += 1
                    else:
                        n_false += 1

                    self.data.append({
                        "input_ids": torch.tensor(
                            ids[:max_seq_len], dtype=torch.long,
                        ),
                        "answer_label": torch.tensor(
                            answer_val, dtype=torch.float32,
                        ),
                        "slot_labels": torch.tensor(
                            slot_labels, dtype=torch.float32,
                        ),
                        "proof_str": str(proofs),
                        "proof_depth": depth,
                    })

        avg_len = total_toks / max(len(self.data), 1)
        total_ex = n_true + n_false
        print(
            f"  Symbolic ProofWriter [{split}]: {len(self.data)} examples "
            f"from {n_theories} theories (skipped {n_skipped} deep)"
        )
        print(
            f"  Vocab: {len(self.vocab)} tokens  |  "
            f"avg seq len: {avg_len:.1f}  |  "
            f"entities anonymised: {len(self.SPECIAL) - 4} IDs available"
        )
        depth_str = "  ".join(
            f"d{d}={depth_counts[d]}" for d in sorted(depth_counts)
        )
        pct_true = 100.0 * n_true / max(total_ex, 1)
        print(
            f"  Labels: {pct_true:.1f}% True  |  Depths: {depth_str}"
        )

    # ── entity helpers ──────────────────────────────────────────────

    @staticmethod
    def _build_entity_map(triples, rules=None, questions=None):
        """Collect unique entities from triples, rules, and questions.

        Entities are mapped to ``[E1]`` … ``[E15]`` in appearance order.
        Sources scanned (in this priority order):
        1. Triple subjects & relational-triple objects.
        2. Rule clauses where relation != "is" → object is an entity.
        3. Question texts — subject (always) and object (in relational Qs).
        """
        entities = []
        seen = set()

        def _add(name):
            if name and name not in seen:
                entities.append(name)
                seen.add(name)

        def _sort_key(k):
            m = re.search(r"\d+", k)
            return int(m.group()) if m else 0

        # ---- 1. triples ------------------------------------------------
        for tkey in sorted(triples, key=_sort_key):
            repr_str = triples[tkey].get("representation", "")
            parts = re.findall(r'"([^"]*)"', repr_str)
            if len(parts) < 4:
                continue

            _add(_normalize_entity(parts[0]))

            relation = parts[1].lower().strip()
            if relation != "is":
                _add(_normalize_entity(parts[2]))

        # ---- 2. rules --------------------------------------------------
        if rules:
            for rkey in sorted(rules, key=_sort_key):
                repr_str = rules[rkey].get("representation", "")
                clauses = re.findall(
                    r'\("([^"]*)" "([^"]*)" "([^"]*)" "([^"]*)"\)',
                    repr_str,
                )
                for ent_field, relation, obj, _pol in clauses:
                    # entity field: usually "someone"/"something"; skip those
                    ent_norm = _normalize_entity(ent_field)
                    if ent_norm not in ("someone", "something", ""):
                        _add(ent_norm)
                    # object field: entity when relation != "is"
                    if relation.lower().strip() != "is":
                        _add(_normalize_entity(obj))

        # ---- 3. questions -----------------------------------------------
        if questions:
            for qkey in sorted(questions, key=_sort_key):
                qtext = questions[qkey].get("question", "")
                for ent in _extract_question_entities(qtext):
                    _add(ent)

        return {
            e: f"[E{i + 1}]"
            for i, e in enumerate(entities[:_MAX_ENTITIES])
        }

    # ── token builders ──────────────────────────────────────────────

    @staticmethod
    def _build_theory_tokens(triples, rules, entity_map, rng=None):
        """Build symbolic token sequence with anonymous entities."""

        def _sort_key(k):
            m = re.search(r"\d+", k)
            return int(m.group()) if m else 0

        # -- facts ------------------------------------------------
        fact_groups = []  # list of token-lists, one per triple
        for tkey in sorted(triples, key=_sort_key):
            repr_str = triples[tkey].get("representation", "")
            parts = re.findall(r'"([^"]*)"', repr_str)
            if len(parts) < 4:
                continue

            subject = _normalize_entity(parts[0])
            relation = parts[1].lower().strip()
            obj_raw = parts[2]

            fact_toks = ["[FACT]"]
            fact_toks.append(entity_map.get(subject, subject))

            if relation == "is":
                prop = _normalize_token(obj_raw)
                if prop and prop not in _STOP_WORDS:
                    fact_toks.append(prop)
            else:
                rel = _normalize_token(relation)
                if rel and rel not in _STOP_WORDS:
                    fact_toks.append(rel)
                obj_entity = _normalize_entity(obj_raw)
                fact_toks.append(entity_map.get(obj_entity, obj_entity))

            fact_groups.append(fact_toks)

        if rng is not None:
            rng.shuffle(fact_groups)

        tokens = []
        for fg in fact_groups:
            tokens.extend(fg)

        # -- rules (entity names anonymised) -----------------------
        for rkey in sorted(rules, key=_sort_key):
            ante, cons = _parse_rule_repr(
                rules[rkey].get("representation", ""),
                entity_map=entity_map,
            )
            if ante or cons:
                tokens.append("[RULE]")
                tokens.extend(ante)
                tokens.append("[IMP]")
                tokens.extend(cons)

        return tokens

    @staticmethod
    def _anon_question(text, entity_map):
        """Parse question text, anonymise entities, detect negation."""
        text = text.lower().strip().rstrip(".")
        words = text.split()

        # 1) detect negation and strip stop-words / copula
        negated = False
        clean = []
        i = 0
        while i < len(words):
            w = words[i]
            if w in ("the", "a", "an"):
                i += 1
                continue
            if w in ("does", "do") and i + 1 < len(words) and words[i + 1] == "not":
                negated = True
                i += 2
                continue
            if w in ("is", "are") and i + 1 < len(words) and words[i + 1] == "not":
                negated = True
                i += 2
                continue
            if w in ("is", "are"):
                i += 1
                continue
            clean.append(w)
            i += 1

        # 2) replace entity substrings (longest first for greedy match)
        text_joined = " ".join(clean)
        for ent in sorted(entity_map, key=len, reverse=True):
            text_joined = text_joined.replace(ent, entity_map[ent])

        # 3) tokenise the result
        result = []
        for w in text_joined.split():
            if w.startswith("[") and w.endswith("]"):
                result.append(w)
            else:
                nw = _normalize_token(w)
                if nw and nw not in _STOP_WORDS:
                    result.append(nw)

        return result, negated

    # ── standard helpers ────────────────────────────────────────────

    def _extract_slot_labels(self, proofs, n_rules, rule_type_map=None):
        """Extract which rule *types* fired from the proof string.

        See ProofWriterDataset._extract_slot_labels for details.
        """
        labels = [0.0] * self.n_slots
        proof_str = str(proofs) if proofs else ""
        for m in re.finditer(r"rule(\d+)", proof_str, re.IGNORECASE):
            rule_key = f"rule{m.group(1)}"
            if self.slot_label_mode == "type" and rule_type_map:
                type_idx = rule_type_map.get(rule_key, NUM_RULE_TYPES - 1)
                if 0 <= type_idx < self.n_slots:
                    labels[type_idx] = 1.0
            else:
                idx = int(m.group(1)) - 1
                if 0 <= idx < self.n_slots:
                    labels[idx] = 1.0
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_answer_rules(n_slots: int = NUM_RULE_TYPES):
        """Each rule type can independently contribute to the answer."""
        return [(i,) for i in range(min(n_slots, NUM_RULE_TYPES))]

    @staticmethod
    def get_rule_names():
        return list(RULE_TYPE_NAMES)


# ── CLUTRR Dataset ─────────────────────────────────────────────────

class CLUTRRDataset(Dataset):
    """
    CLUTRR kinship reasoning dataset loader.

    Download from: https://github.com/facebookresearch/clutrr
    Expected format: CSV with columns:
      - "story" or "clean_story": the family narrative
      - "query": the relation query
      - "target": ground-truth kinship relation

    Place files as:  {data_dir}/train.csv, {data_dir}/test.csv, etc.
    """

    KINSHIP_PREDICATES = [
        "mother", "father", "sister", "brother",
        "daughter", "son", "grandmother", "grandfather",
    ]

    KINSHIP_KEYWORDS = {
        0: ["mother", "mom", "mommy"],
        1: ["father", "dad", "daddy"],
        2: ["sister", "sis"],
        3: ["brother", "bro"],
        4: ["daughter"],
        5: ["son"],
        6: ["grandmother", "grandma", "granny"],
        7: ["grandfather", "grandpa", "granddad"],
    }

    def __init__(
        self,
        data_dir,
        split="train",
        max_seq_len=256,
        n_slots=7,
        vocab=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_slots = n_slots
        self.vocab = vocab or SimpleVocab()

        candidates = [
            os.path.join(data_dir, f"{split}.csv"),
            os.path.join(data_dir, split, "data.csv"),
            os.path.join(data_dir, f"{split}_data.csv"),
        ]
        data_path = next((c for c in candidates if os.path.exists(c)), None)

        if data_path is None:
            raise FileNotFoundError(
                f"CLUTRR data not found. Searched:\n"
                + "\n".join(f"  - {c}" for c in candidates)
                + f"\n\nDownload from: https://github.com/facebookresearch/clutrr"
                + f"\nPlace CSV files in: {data_dir}/"
            )

        self.data = []
        self.relation_to_idx = {r: i for i, r in enumerate(self.KINSHIP_PREDICATES)}
        self.n_classes = len(self.KINSHIP_PREDICATES)

        n_known = 0
        n_unknown = 0
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                story = row.get("story", row.get("clean_story", ""))
                query = row.get("query", "")
                target = row.get("target", row.get("relation", "")).lower().strip()

                text = story + " [SEP] " + query if query else story
                input_ids = self.vocab.encode(text, max_seq_len)

                # Multi-class kinship label
                relation_idx = self.relation_to_idx.get(target, -1)
                if relation_idx >= 0:
                    n_known += 1
                else:
                    relation_idx = 0  # fallback
                    n_unknown += 1

                # Binary answer: 1.0 if relation is correctly identified
                # (kept for backward compat; real evaluation uses relation_label)
                answer = 1.0

                slot_labels = self._extract_predicates(story)

                self.data.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "answer_label": torch.tensor(answer, dtype=torch.float32),
                    "slot_labels": torch.tensor(slot_labels, dtype=torch.float32),
                    "relation_label": torch.tensor(relation_idx, dtype=torch.long),
                    "target_relation": target,
                })

        print(f"  CLUTRR [{split}]: {len(self.data)} examples loaded "
              f"({n_known} known relations, {n_unknown} unknown)")

    def _extract_predicates(self, story):
        """Extract kinship predicates mentioned in the story."""
        labels = [0.0] * self.n_slots
        story_lower = story.lower()
        for slot_idx, keywords in self.KINSHIP_KEYWORDS.items():
            if slot_idx < self.n_slots:
                if any(kw in story_lower for kw in keywords):
                    labels[slot_idx] = 1.0
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_answer_rules():
        """Kinship composition rules.

        Encodes common kinship inferences:
          mother + mother -> grandmother  (slot 0, 0 -> slot 6)
          father + father -> grandfather  (slot 1, 1 -> slot 7)
          mother + father -> grandmother  (slot 0, 1 -> slot 6)
        """
        return [(0, 0), (1, 1), (0, 1)]

    @staticmethod
    def get_rule_names():
        return [
            "IsMother", "IsFather", "IsSister", "IsBrother",
            "IsDaughter", "IsSon", "IsGrandmother", "IsGrandfather",
        ]


# ── Data Loader Helper ──────────────────────────────────────────────

def collate_fn(batch):
    """Collate function for DataLoader."""
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "answer_label": torch.stack([b["answer_label"] for b in batch]),
        "slot_labels": torch.stack([b["slot_labels"] for b in batch]),
    }
    # Optional fields from SymbolicProofWriter / CLUTRR
    if "proof_str" in batch[0]:
        result["proof_str"] = [b["proof_str"] for b in batch]
    if "proof_depth" in batch[0]:
        result["proof_depth"] = torch.tensor(
            [b["proof_depth"] for b in batch], dtype=torch.long
        )
    if "relation_label" in batch[0]:
        result["relation_label"] = torch.stack(
            [b["relation_label"] for b in batch]
        )
    return result


class CurriculumSampler:
    """
    Depth-aware sampler for curriculum learning.

    Given a SymbolicProofWriterDataset, filters examples by proof depth.
    Call ``set_max_depth(d)`` to change which examples are included.
    Wraps as a Subset for use with DataLoader.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        # Pre-index examples by depth
        self.depth_indices = {}  # depth → list of indices
        base = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        for i in range(len(base)):
            d = base.data[i].get("proof_depth", 0)
            self.depth_indices.setdefault(d, []).append(i)
        self._current_max_depth = max(self.depth_indices.keys())
        self._current_indices = list(range(len(base)))

    def set_max_depth(self, max_depth: int):
        """Update which examples are included (depth ≤ max_depth)."""
        self._current_max_depth = max_depth
        self._current_indices = []
        for d, idxs in self.depth_indices.items():
            if d <= max_depth:
                self._current_indices.extend(idxs)
        self._current_indices.sort()

    def get_subset(self):
        """Return a Subset of the dataset with current depth filter."""
        from torch.utils.data import Subset
        base = self.dataset.dataset if hasattr(self.dataset, 'dataset') else self.dataset
        return Subset(base, self._current_indices)

    @property
    def current_depth(self):
        return self._current_max_depth

    @property
    def n_examples(self):
        return len(self._current_indices)

    def summary(self):
        from collections import Counter
        base = self.dataset.dataset if hasattr(self.dataset, 'dataset') else self.dataset
        counts = Counter()
        for i in self._current_indices:
            counts[base.data[i].get("proof_depth", 0)] += 1
        return dict(sorted(counts.items()))


def get_dataloaders(
    dataset_name="synthetic",
    batch_size=32,
    n_train=800,
    n_val=200,
    data_dir="data",
    max_examples=0,
    encoding="text",
    **kwargs,
):
    """
    Build train/val DataLoaders.

    Args:
        dataset_name: "synthetic", "proofwriter", or "clutrr"
        batch_size: batch size
        n_train: number of training examples (synthetic only)
        n_val: number of validation examples (synthetic only)
        data_dir: root directory for ProofWriter / CLUTRR data
        max_examples: subsample training data to this size (0=all)
        encoding: "text" (raw NL) or "symbolic" (structured tokens).
                  Only affects ProofWriter.
    """
    if dataset_name == "synthetic":
        train_ds = SyntheticRulesDataset(n_examples=n_train, seed=42, **kwargs)
        val_ds = SyntheticRulesDataset(n_examples=n_val, seed=123, **kwargs)
    elif dataset_name == "proofwriter":
        pw_dir = os.path.join(data_dir, "proofwriter")
        vocab = SimpleVocab()
        pw_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("max_seq_len", "n_slots", "max_depth", "subset",
                              "shuffle_facts", "slot_label_mode")}

        DatasetCls = (SymbolicProofWriterDataset
                      if encoding == "symbolic" else ProofWriterDataset)
        train_ds = DatasetCls(pw_dir, split="train", vocab=vocab, **pw_kwargs)
        vocab.freeze()
        val_ds = DatasetCls(pw_dir, split="val", vocab=vocab, **pw_kwargs)
    elif dataset_name == "clutrr":
        cl_dir = os.path.join(data_dir, "clutrr")
        vocab = SimpleVocab()
        cl_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("max_seq_len", "n_slots")}
        train_ds = CLUTRRDataset(cl_dir, split="train", vocab=vocab, **cl_kwargs)
        vocab.freeze()
        val_ds = CLUTRRDataset(cl_dir, split="test", vocab=vocab, **cl_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Subsample training data if requested
    if max_examples and max_examples > 0 and len(train_ds) > max_examples:
        from torch.utils.data import Subset
        indices = torch.randperm(len(train_ds))[:max_examples].tolist()
        train_ds = Subset(train_ds, indices)
        print(f"  Subsampled train to {max_examples} examples")

    # Also cap val to max 2000 for speed
    if max_examples and max_examples > 0 and len(val_ds) > max(2000, max_examples // 2):
        from torch.utils.data import Subset
        val_cap = max(2000, max_examples // 2)
        indices = torch.randperm(len(val_ds))[:val_cap].tolist()
        val_ds = Subset(val_ds, indices)
        print(f"  Subsampled val to {val_cap} examples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    return train_loader, val_loader
