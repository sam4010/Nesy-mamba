"""Test proof parser."""
from nesy_mamba.proof_parser import ProofParser

tests = [
    ('[(triple2)]', 0, set(), {2}),
    ('[(((triple7 triple5) -> rule3))]', 1, {3}, {7, 5}),
    ('[(((triple8 ((triple7 triple5) -> rule3)) -> rule4))]', 2, {3, 4}, {8, 7, 5}),
    ('[(((triple4 ((triple8 ((triple7 triple5) -> rule3)) -> rule4) triple5) -> rule2))]', 3, {3, 4, 2}, {4, 8, 7, 5}),
    ('[(((triple10) -> rule7) OR ((triple4) -> rule7))]', 1, {7}, {10}),
]

all_ok = True
for idx, (proof_str, expected_depth, expected_rules, expected_triples) in enumerate(tests):
    tree = ProofParser.parse(proof_str)
    depth_ok = tree.max_depth == expected_depth
    rules_ok = tree.get_rules_used() == expected_rules
    triples_ok = tree.get_triples_used() == expected_triples
    ok = depth_ok and rules_ok and triples_ok
    
    status = "OK" if ok else "FAIL"
    print(f"Test {idx}: {status}")
    if not ok:
        all_ok = False
        print(f"  Input: {proof_str}")
        if not depth_ok:
            print(f"  depth: got {tree.max_depth}, expected {expected_depth}")
        if not rules_ok:
            print(f"  rules: got {tree.get_rules_used()}, expected {expected_rules}")
        if not triples_ok:
            print(f"  triples: got {tree.get_triples_used()}, expected {expected_triples}")
        print(f"  primary: {tree.primary}")
        print(f"  alts: {len(tree.alternatives)}")

print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
