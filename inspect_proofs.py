"""Inspect ProofWriter JSON proof structure."""
import json
import os

pw_dir = os.path.join("nesy_mamba", "data", "proofwriter", "CWA", "depth-3ext-NatLang")
with open(os.path.join(pw_dir, "meta-train.jsonl"), "r") as f:
    theories = [json.loads(line) for i, line in enumerate(f) if i < 3]

for ti, theory in enumerate(theories):
    print(f"{'='*70}")
    print(f"THEORY {ti}")
    print(f"{'='*70}")
    print("Keys:", list(theory.keys()))
    
    print("\n--- TRIPLES (first 3) ---")
    for k in list(theory.get("triples", {}).keys())[:3]:
        print(f"  {k}: {theory['triples'][k]}")
    
    print(f"\n--- RULES ({len(theory.get('rules', {}))} total, first 3) ---")
    for k in list(theory.get("rules", {}).keys())[:3]:
        print(f"  {k}: {theory['rules'][k]}")
    
    print("\n--- QUESTIONS (first 5) ---")
    for k in list(theory.get("questions", {}).keys())[:5]:
        q = theory["questions"][k]
        print(f"  {k}:")
        print(f"    question: {q.get('question', '')}")
        print(f"    answer:   {q.get('answer', '')}")
        print(f"    QDep:     {q.get('QDep', '')}")
        proofs = q.get("proofs", "")
        print(f"    proofs:   {proofs}")
        print()
    print()
