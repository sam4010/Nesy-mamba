"""Debug proof parser step by step."""
from nesy_mamba.proof_parser import _tokenize, _TokenStream, _parse_atom, ProofParser

proof_str = "[(((triple7 triple5) -> rule3))]"
tokens = _tokenize(proof_str)
print("Tokens:", tokens)

# Manually trace
import nesy_mamba.proof_parser as pp

# Patch _parse_atom with debug
_orig = pp._parse_atom

def _debug_parse(stream, parent_list=None, depth=0):
    indent = "  " * depth
    tok = stream.peek()
    print(f"{indent}_parse_atom pos={stream.pos} peek={tok} parent_has={len(parent_list) if parent_list is not None else 'None'}")
    
    if tok is None:
        return None
    if tok.startswith("triple"):
        stream.consume()
        node = pp.TripleNode(triple_id=int(tok[6:]), depth=0)
        print(f"{indent}  -> {node}")
        return node
    if tok == "(":
        stream.consume()
        antecedents = []
        while stream.peek() not in (None, "->", ")"):
            child = _debug_parse(stream, parent_list=antecedents, depth=depth+1)
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
                node = pp.RuleApplication(rule_id=rid, antecedents=antecedents, depth=max_d+1)
                print(f"{indent}  -> RuleApp(rule{rid}, ante={[repr(a) for a in antecedents]})")
                return node
        
        if stream.peek() == ")":
            stream.consume()
        
        print(f"{indent}  grouping paren: {[repr(a) for a in antecedents]}")
        if parent_list is not None and len(antecedents) > 1:
            for extra in antecedents[1:]:
                parent_list.append(extra)
            print(f"{indent}  spliced {len(antecedents)-1} extra to parent")
            return antecedents[0] if antecedents else None
        if len(antecedents) == 1:
            return antecedents[0]
        if antecedents:
            return antecedents[0]
        return None
    
    stream.consume()
    return None

stream = _TokenStream(tokens)
result = _debug_parse(stream, depth=0)
print(f"\nResult: {result}")
if hasattr(result, 'antecedents'):
    print(f"Antecedents: {result.antecedents}")
