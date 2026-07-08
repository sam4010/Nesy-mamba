"""Analyze ProofWriter rule structures to design rule type taxonomy."""
import json, re
from collections import Counter

f = open(r'nesy_mamba\data\proofwriter\CWA\depth-5\meta-train.jsonl', 'r', encoding='utf-8')

type_counts = Counter()
n_ante_counts = Counter()
n_rules_per_theory = Counter()
total_rules = 0
CLAUSE_RE = re.compile(r'\("([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\)')

for i, line in enumerate(f):
    line = line.strip()
    if not line: continue
    theory = json.loads(line)
    rules = theory.get('rules', {})
    n_rules_per_theory[len(rules)] += 1
    
    for rkey, rval in rules.items():
        rep = rval.get('representation', '')
        total_rules += 1
        
        if '->' not in rep:
            type_counts['NO_ARROW'] += 1
            continue
        
        ante_str, cons_str = rep.split('->', 1)
        
        ante_clauses = CLAUSE_RE.findall(ante_str)
        cons_clauses = CLAUSE_RE.findall(cons_str)
        
        n_ante = len(ante_clauses)
        n_ante_counts[n_ante] += 1
        
        ante_types = []
        for ent, rel, obj, pol in ante_clauses:
            if rel.lower().strip() == 'is':
                ante_types.append('prop')
            else:
                ante_types.append('rel')
        
        cons_types = []
        for ent, rel, obj, pol in cons_clauses:
            if rel.lower().strip() == 'is':
                cons_types.append('prop')
            else:
                cons_types.append('rel')
        
        ante_key = '+'.join(sorted(ante_types)) if ante_types else 'none'
        cons_key = '+'.join(sorted(cons_types)) if cons_types else 'none'
        type_key = f'{ante_key}({n_ante})->{cons_key}'
        type_counts[type_key] += 1

f.close()
print(f'Total rules: {total_rules}')
print(f'\nRules per theory: {dict(sorted(n_rules_per_theory.items()))}')
print(f'\nAntecedent count distribution: {dict(sorted(n_ante_counts.items()))}')
print(f'\nRule type distribution:')
for t, c in type_counts.most_common(20):
    print(f'  {t:40s} {c:5d} ({100*c/total_rules:.1f}%)')
