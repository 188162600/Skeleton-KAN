"""Source-only complexity annotations, relocated without rule changes."""
from collections import defaultdict
from functools import lru_cache

@lru_cache(None)
def symbolic_family(text):
    # Symbolic source metadata only: never used to fit or initialize a model.
    import sympy as sp
    z,t=sp.symbols('z t', real=True)
    e=sp.sympify(text,locals={'z':z})
    if e.free_symbols-{z}:return None
    def affine(e,var):return e.is_polynomial(var) and sp.Poly(e,var).degree()<=1
    if affine(e,z):return 'affine'
    if e.is_polynomial(z) and sp.Poly(e,z).degree()==2:return 'square'
    for power in e.atoms(sp.Pow):
        if power.exp == -2 and affine(power.base,z):
            rest=e.xreplace({power:t})
            if not rest.has(z) and affine(rest,t):return 'inverse_square'
    num,den=sp.fraction(sp.cancel(e))
    if affine(num,z) and affine(den,z) and sp.Poly(den,z).degree()==1:return 'reciprocal'
    for op in (sp.sin,sp.cos,sp.tan):
        atoms=list(e.atoms(op))
        if len(atoms)==1 and affine(atoms[0].args[0],z):
            rest=e.xreplace({atoms[0]:t})
            if not rest.has(z) and affine(rest,t):return str(op)
            if not rest.has(z) and rest.is_polynomial(t):
                poly=sp.Poly(rest,t)
                if op in (sp.sin,sp.cos) and poly.degree()==2 and poly.nth(1)==0:
                    return str(op)+'²'
    return None


def score(edge):
    p = str(edge.get('primitive', 'x')).replace(' ', '')
    if p in ('x', 'identity', 'z'): return 3, 'affine'
    if p in ('1/x', '1/z', 'x^-1', 'x**-1', 'x**(-1)'): return 15, 'reciprocal'
    if p in ('1/x^2','1/z^2','1/x**2','1/z**2','x^-2','z^-2','x**-2','z**-2','x**(-2)','z**(-2)','inverse_square','pow(-2)'): return 15,'inverse_square'
    if p in ('x^2', 'x**2', 'z**2','x²'): return 9,'square'
    if p in ('sin', 'cos', 'tan','sin²','cos²'): return 9,p
    if p=='symbolic':
        text=edge.get('symbolic_expression','')
        family=symbolic_family(text) if text else None
        if family:return {'affine':3,'reciprocal':15,'inverse_square':15}.get(family,9),family
        return 9,'provisional:'+text
    return 9, 'provisional:'+p


def combine(labels, multiplied=False):
    non = [s for s in labels if s[0] != 3]
    if len(labels) == 1: return labels[0]
    if not non and not multiplied: return 3, 'affine'
    if len(non) == 1 and not multiplied: return non[0]
    return 9, 'provisional:composed'


def annotated(spec, structural):
    """Same affine-role, identity-flattening and unary-collapse rules as v2."""
    nodes = {str(n['id']):n for n in spec['nodes']}
    roles = {}
    declared = set(spec['variables'])
    def role(source):
        indices = tuple(int(v) for v in source.get('variable_indices', ()))
        if not indices: return None
        assert set(source.get('variables', ())).issubset(declared)
        key = ('raw', indices[0]) if source.get('source_kind')=='raw' and len(indices)==1 else (
            'affine', indices, tuple(round(float(v),12) for v in source.get('coefficients',())),
            round(float(source.get('constant',0)),12))
        if key not in roles: roles[key] = len(roles)
        return roles[key]
    def values(n):
        return [s for _,s in n['raw']] + [c['incoming'] for c in n['children']] + [s for c in n['children'] for s in values(c)]
    def key(n):
        return (n['op'], len(n['raw']), tuple(sorted(key(c) for c in n['children'])))
    def visit(identifier):
        n=nodes[identifier];op=n['operator'];raw=[];children=[];deps=set()
        for s in n.get('raw_sources',[]):
            r=role(s)
            if r is not None: raw.append((r,score(s.get('edge',{}))));deps.add(r)
        for s in n.get('state_sources',[]):
            child,cd,post,post_unary=visit(str(s['state']));deps.update(cd)
            if child is None: continue
            incoming=combine([post,score(s.get('edge',{}))])
            incoming_unary=post_unary or structural.edge_route(s.get('edge',{}))=='unary'
            if len(cd)<=1:
                if cd:
                    labels=values(child)+[incoming]
                    # A nontrivial collapsed unary computation is one generic map.
                    q=combine(labels, child['op']=='*' and len(child['raw'])>1)
                    raw.append((next(iter(cd)), q))
                continue
            if child['op']==op and not incoming_unary:
                raw.extend(child['raw']);children.extend(child['children'])
            else:
                child['incoming']=incoming;child['incoming_unary']=incoming_unary;children.append(child)
        grouped=defaultdict(list)
        for r,q in raw:grouped[r].append(q)
        raw=[(r,combine(v,op=='*' and len(v)>1)) for r,v in sorted(grouped.items())]
        children.sort(key=key)
        if not raw and not children:return None,deps,(3,'affine'),False
        if not raw and len(children)==1:
            child=children[0];return child,deps,child.pop('incoming'),child.pop('incoming_unary')
        return dict(op=op,raw=raw,children=children),deps,(3,'affine'),False
    oid=str(spec['output_sources'][0]);root,_,post,_=visit(oid)
    out=next((e for e in spec.get('output_edges',[]) if str(e['source'])==oid),{})
    output=combine([post,score(out)])
    rich,_=structural.rich_from_spec(spec,collapse_univariate_chains=True)
    def rk(n):return(n.operator,len(n.raw_ports),tuple(sorted(rk(c.node) for c in n.children)))
    assert key(root)==rk(rich),('annotation topology drift',key(root),rk(rich))
    return root,output,values(root)+[output]

