"""Test the rule type classifier."""
from nesy_mamba.data_utils import classify_rule, classify_theory_rules, RULE_TYPE_NAMES, NUM_RULE_TYPES

print("Rule types:", RULE_TYPE_NAMES)
print("NUM_RULE_TYPES:", NUM_RULE_TYPES)

# Test classify_rule on representative examples
tests = [
    ('((("something" "is" "big" "+")) -> ("something" "is" "kind" "+"))',
     'prop(1)->prop = slot 0'),
    ('((("something" "is" "big" "+") ("something" "is" "young" "+")) -> ("something" "is" "kind" "+"))',
     'prop+prop(2)->prop = slot 1'),
    ('((("something" "sees" "tiger" "+")) -> ("something" "sees" "rabbit" "+"))',
     'rel(1)->rel = slot 2'),
    ('((("something" "is" "big" "+")) -> ("something" "sees" "lion" "+"))',
     'prop(1)->rel = slot 3'),
    ('((("something" "sees" "tiger" "+")) -> ("something" "is" "kind" "+"))',
     'rel(1)->prop = slot 4'),
    ('((("something" "is" "big" "+") ("something" "sees" "tiger" "+")) -> ("something" "sees" "rabbit" "+"))',
     'mixed(2)->rel = slot 5'),
    ('((("something" "sees" "tiger" "+") ("something" "sees" "lion" "+")) -> ("something" "is" "kind" "+"))',
     'rel+rel(2)->prop = slot 6'),
]

for rep, expected in tests:
    t = classify_rule(rep)
    print(f"  {expected:35s} -> slot {t} ({RULE_TYPE_NAMES[t]})")

# Test on real data
import json
with open(r"nesy_mamba\data\proofwriter\CWA\depth-5\meta-train.jsonl", "r") as f:
    theory = json.loads(f.readline())
    rules = theory["rules"]
    mapping = classify_theory_rules(rules)
    print(f"\nTheory has {len(rules)} rules:")
    for rkey in sorted(mapping, key=lambda k: int(k[4:])):
        rep = rules[rkey]["representation"]
        t = mapping[rkey]
        print(f"  {rkey}: type={t} ({RULE_TYPE_NAMES[t]}) | {rep[:80]}...")
