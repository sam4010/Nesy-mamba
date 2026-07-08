"""Inspect ProofWriter deep proofs."""
import json, os

pw_dir = os.path.join("nesy_mamba", "data", "proofwriter", "CWA", "depth-3ext-NatLang")
with open(os.path.join(pw_dir, "meta-train.jsonl"), "r") as f:
    found = 0
    for i, line in enumerate(f):
        theory = json.loads(line)
        for qk, qv in theory["questions"].items():
            depth = qv.get("QDep", 0)
            if depth >= 3 and found < 5:
                proof = qv["proofs"]
                q = qv["question"]
                a = qv["answer"]
                print(f"Theory {i}, {qk} (depth={depth}):")
                print(f"  Q: {q}")
                print(f"  A: {a}")
                print(f"  Proofs: {proof}")
                
                # Also show which rules are involved
                import re
                rules = re.findall(r"rule\d+", str(proof))
                triples = re.findall(r"triple\d+", str(proof))
                print(f"  Rules used: {set(rules)}")
                print(f"  Triples used: {set(triples)}")
                print()
                found += 1
        if found >= 5:
            break

# Also check: are there "False" answers at depth>=1 with proofs?
print("\n=== FALSE ANSWERS WITH PROOFS ===")
with open(os.path.join(pw_dir, "meta-train.jsonl"), "r") as f:
    found2 = 0
    for i, line in enumerate(f):
        theory = json.loads(line)
        for qk, qv in theory["questions"].items():
            depth = qv.get("QDep", 0)
            if depth >= 1 and not qv.get("answer", True) and found2 < 3:
                proof = qv["proofs"]
                q = qv["question"]
                print(f"Theory {i}, {qk} (depth={depth}, answer=False):")
                print(f"  Q: {q}")
                print(f"  Proofs: {proof}")
                print()
                found2 += 1
        if found2 >= 3:
            break
