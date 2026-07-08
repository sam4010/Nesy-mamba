"""
Proof Tree Parser for ProofWriter.

Parses proof strings from ProofWriter JSON into structured proof trees,
enabling:
  1. Per-example dynamic rule extraction for L_rule
  2. Proof-depth correlation analysis with slot firing order
  3. Compositional generalization evaluation

Proof string grammar (recursive):
    proof     ::= "[" "(" chain ")" "]"
    chain     ::= "(" args "->" rule_ref ")" | triple_ref | chain " OR " chain
    args      ::= (chain | triple_ref)+
    triple_ref::= "triple" INT
    rule_ref  ::= "rule" INT
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Proof Tree Nodes ────────────────────────────────────────────────

@dataclass
class ProofNode:
    """Base class for proof tree nodes."""
    depth: int = 0


@dataclass
class TripleNode(ProofNode):
    """Leaf: references a ground-truth triple (fact)."""
    triple_id: int = 0

    def __repr__(self):
        return f"triple{self.triple_id}"


@dataclass
class RuleApplication(ProofNode):
    """Internal node: a rule applied to antecedent sub-proofs."""
    rule_id: int = 0
    antecedents: list[ProofNode] = field(default_factory=list)

    def __repr__(self):
        ante_str = " ".join(repr(a) for a in self.antecedents)
        return f"(({ante_str}) -> rule{self.rule_id})"


@dataclass
class ProofTree:
    """A complete proof (possibly with OR-alternatives)."""
    alternatives: list[ProofNode] = field(default_factory=list)

    @property
    def primary(self) -> Optional[ProofNode]:
        return self.alternatives[0] if self.alternatives else None

    @property
    def max_depth(self) -> int:
        return max((a.depth for a in self.alternatives), default=0)

    def get_rules_used(self) -> set[int]:
        if not self.primary:
            return set()
        return _collect_rules(self.primary)

    def get_triples_used(self) -> set[int]:
        if not self.primary:
            return set()
        return _collect_triples(self.primary)

    def get_rule_chain(self) -> list[int]:
        """Rules in bottom-up order (deepest first) from primary proof."""
        if not self.primary:
            return []
        chain: list[int] = []
        _collect_rule_chain(self.primary, chain)
        return chain

    def get_antecedent_rules_for_answer(self) -> list[tuple[int, ...]]:
        """Per-example antecedent rule tuples for L_rule.

        Each tuple contains 0-based rule indices forming one
        implication step in the proof chain.
        """
        if not self.primary:
            return []
        rules: list[tuple[int, ...]] = []
        _extract_implication_rules(self.primary, rules)
        return rules


# ── Recursive Helpers ───────────────────────────────────────────────

def _collect_rules(node: ProofNode) -> set[int]:
    if isinstance(node, TripleNode):
        return set()
    if isinstance(node, RuleApplication):
        result = {node.rule_id}
        for a in node.antecedents:
            result |= _collect_rules(a)
        return result
    return set()


def _collect_triples(node: ProofNode) -> set[int]:
    if isinstance(node, TripleNode):
        return {node.triple_id}
    if isinstance(node, RuleApplication):
        result: set[int] = set()
        for a in node.antecedents:
            result |= _collect_triples(a)
        return result
    return set()


def _collect_rule_chain(node: ProofNode, chain: list[int]):
    if isinstance(node, RuleApplication):
        for a in node.antecedents:
            _collect_rule_chain(a, chain)
        chain.append(node.rule_id)


def _extract_implication_rules(
    node: ProofNode, rules: list[tuple[int, ...]]
):
    if isinstance(node, RuleApplication):
        for a in node.antecedents:
            _extract_implication_rules(a, rules)
        ante_rules = []
        for a in node.antecedents:
            if isinstance(a, RuleApplication):
                ante_rules.append(a.rule_id - 1)
        this_rule = node.rule_id - 1
        if ante_rules:
            rules.append(tuple(ante_rules + [this_rule]))
        else:
            rules.append((this_rule,))


# ── Tokenizer-based Parser ─────────────────────────────────────────

_TOKEN_RE = re.compile(r"\(|\)|->|OR|triple\d+|rule\d+")


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s)


class _TokenStream:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self) -> Optional[str]:
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def done(self) -> bool:
        return self.pos >= len(self.tokens)


def _parse_atom(stream: _TokenStream, parent_list: list[ProofNode] | None = None) -> Optional[ProofNode]:
    """Parse one proof atom.  When *parent_list* is provided, bare
    grouping parens ``(a b c)`` splice their children directly into
    the parent instead of discarding all but the first.
    """
    tok = stream.peek()
    if tok is None:
        return None

    # Triple reference
    if tok.startswith("triple"):
        stream.consume()
        return TripleNode(triple_id=int(tok[6:]), depth=0)

    # Parenthesized expression
    if tok == "(":
        stream.consume()

        antecedents: list[ProofNode] = []
        while stream.peek() not in (None, "->", ")"):
            child = _parse_atom(stream, parent_list=antecedents)
            if child:
                antecedents.append(child)

        if stream.peek() == "->":
            stream.consume()
            rule_tok = stream.consume()
            if rule_tok and rule_tok.startswith("rule"):
                rid = int(rule_tok[4:])
                if stream.peek() == ")":
                    stream.consume()
                max_d = max((a.depth for a in antecedents), default=0)
                return RuleApplication(
                    rule_id=rid, antecedents=antecedents, depth=max_d + 1
                )

        # Close paren without arrow — bare grouping parens
        if stream.peek() == ")":
            stream.consume()

        # Splice children into parent's list (if available)
        if parent_list is not None and len(antecedents) > 1:
            # Return the first, push extras into parent
            for extra in antecedents[1:]:
                parent_list.append(extra)
            return antecedents[0] if antecedents else None

        if len(antecedents) == 1:
            return antecedents[0]
        if antecedents:
            return antecedents[0]
        return None

    # Unknown token — skip
    stream.consume()
    return None


class ProofParser:
    @staticmethod
    def parse(proof_str: str) -> ProofTree:
        if not proof_str or proof_str in ("", "[]", "None", "none"):
            return ProofTree()

        proof_str = str(proof_str).strip()
        tokens = _tokenize(proof_str)
        if not tokens:
            return ProofTree()

        alt_groups = _split_by_or(tokens)
        nodes = []
        for alt_tokens in alt_groups:
            stream = _TokenStream(alt_tokens)
            try:
                node = _parse_atom(stream)
                if node:
                    nodes.append(node)
            except (ValueError, IndexError):
                pass

        return ProofTree(alternatives=nodes)


def _split_by_or(tokens: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    depth = 0

    for tok in tokens:
        if tok == "(":
            depth += 1
            current.append(tok)
        elif tok == ")":
            depth -= 1
            current.append(tok)
        elif tok == "OR" and depth <= 1:
            if current:
                groups.append(current)
            current = []
        else:
            current.append(tok)

    if current:
        groups.append(current)
    return groups if groups else [tokens]


# ── Batch Processing ────────────────────────────────────────────────

def parse_proof_batch(proof_strings: list[str]) -> list[ProofTree]:
    return [ProofParser.parse(p) for p in proof_strings]


def extract_dynamic_rules(
    proof_trees: list[ProofTree],
    n_slots: int,
) -> list[list[tuple[int, ...]]]:
    """Extract per-example dynamic rules for L_rule.

    Returns list of lists of tuples (one per example),
    where each tuple contains 0-based slot indices forming
    an antecedent chain.
    """
    batch_rules = []
    for tree in proof_trees:
        if tree.primary is None:
            batch_rules.append([])
            continue
        rules = tree.get_antecedent_rules_for_answer()
        valid = [
            tuple(idx for idx in r if 0 <= idx < n_slots)
            for r in rules
        ]
        batch_rules.append([v for v in valid if v])
    return batch_rules


def get_proof_rule_chain(proof_str: str) -> list[int]:
    """Get ordered rule chain (0-based, bottom-up) from proof string."""
    tree = ProofParser.parse(proof_str)
    return [r - 1 for r in tree.get_rule_chain()]
