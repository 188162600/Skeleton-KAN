"""Original-expression ASTs and ordered TED; phi does not enter clustering."""
from dataclasses import dataclass
import numpy as np
import sympy as sp
from .semantics import declared_inputs,parse_declared_expression
from .vendor_api import GraphInput

@dataclass
class ExprTree:
    name: str
    children: list

    def signature(self):
        return (self.name, tuple(c.signature() for c in self.children))

    def payload(self):
        return {'label': self.name, 'children': [c.payload() for c in self.children]}


def original_tree(row):
    from . import semantics as input_semantics
    from . import cache as pc
    identity = {'expression': row['analysis_expression'], 'inputs': declared_inputs(row),
                'semantics': pc.fingerprint(__file__, input_semantics.__file__), 'sympy': sp.__version__}
    cached = pc.get('original_ast_v1', identity)
    if cached is not None:
        return pc.tree_from_payload(cached)
    expr, inputs = parse_declared_expression(row['analysis_expression'], declared_inputs(row))
    names = {x: f'input:{i}' for i, x in enumerate(inputs)}
    def visit(x):
        if x in names:
            return ExprTree(names[x], [])
        if not x.free_symbols.intersection(inputs):
            # Compound noninput expressions are constants, never input roles.
            return ExprTree('constant:' + sp.sstr(x), [])
        op = 'sum' if x.is_Add else 'prod' if x.is_Mul else x.func.__name__
        children = [visit(c) for c in x.args]
        if x.is_Add or x.is_Mul:
            children.sort(key=lambda t: t.signature())
        return ExprTree(op, children)
    tree = visit(expr)
    pc.put('original_ast_v1', identity, tree.payload())
    return tree


def ordered_distance(a, b):
    from apted import APTED
    from . import cache as pc
    identity = pc.distance_identity(a, b)
    cached = pc.get('apted_1.0.3_ordered_unit_cost_v1', identity)
    if cached is not None:
        return float(cached)
    distance = float(APTED(a, b).compute_edit_distance())
    pc.put('apted_1.0.3_ordered_unit_cost_v1', identity, distance)
    return distance


def groups_at(folder, k):
    from scipy.cluster.hierarchy import cut_tree
    with np.load(folder/'original_ted.npz') as data:
        distances, hierarchy = data['distances'], data['linkage']
    if not 1 <= k <= len(distances):
        raise ValueError('K must be between one and the number of source trees')
    labels = cut_tree(hierarchy, n_clusters=[k]).reshape(-1) if len(distances)>1 else np.zeros(1,dtype=int)
    groups = [np.flatnonzero(labels == j).tolist() for j in range(k)]
    assert len(groups) == k and all(groups) and sum(map(len, groups)) == len(distances)
    return groups, distances


def graph_from_tree(tree, labels):
    from scipy.sparse.csgraph import shortest_path
    from . import cache as pc
    identity = pc.tree_identity(tree)
    cached = pc.get('undirected_tree_shortest_paths_v1', identity)
    if cached is not None:
        # Labels/one-hot coordinates always come from THIS source fold.
        return GraphInput(np.asarray([labels[x] for x in cached['labels']], dtype=float), np.asarray(cached['distances'], dtype=float))
    features = []; edges = []
    names = []
    def visit(t, parent=None):
        i = len(features)
        features.append(labels[t.name])
        names.append(t.name)
        if parent is not None:
            edges.append((parent, i))
        for c in t.children:
            visit(c, i)
    visit(tree)
    a = np.zeros((len(features), len(features)))
    for i,j in edges:
        a[i,j] = a[j,i] = 1
    structure = shortest_path(a, directed=False, unweighted=True)
    pc.put('undirected_tree_shortest_paths_v1', identity, {'labels': names, 'distances': structure.tolist()})
    return GraphInput(np.asarray(features, dtype=float), structure)

