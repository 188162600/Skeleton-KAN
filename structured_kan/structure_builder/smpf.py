"""Fixed SMPF-inspired coordinate-interaction sweep; no target equation is read.

Coordinate mode keeps variable groups fixed and replaces only the primitive
maps with kernels. Legacy full-affine mode remains explicit for replay.
Neither mode runs native SMPF's genetic search.
"""
import hashlib
import math
import random

METHOD = 'smpf_structure'
TOPOLOGY_SEED = 20260910


def architecture_sweep(*, input_mode='full_affine'):
    if input_mode not in ('full_affine', 'coordinate'):
        raise ValueError(input_mode)
    rows = []
    for G in (10, 30):
        for L in (2, 4, 8):
            for numerator in (1, 2, 3):
                rows.append(dict(id=f'SMPF{len(rows)+1:02d}', hidden_sums=L,
                    route_fraction_numerator=numerator, route_fraction_denominator=3,
                    G=G, topology_seed=TOPOLOGY_SEED))
                if input_mode == 'coordinate':
                    rows[-1]['input_mode'] = 'coordinate'
    return rows


def layout(input_dim, architecture):
    assert type(input_dim) is int and input_dim >= 1
    L = architecture['hidden_sums']
    n, den = architecture['route_fraction_numerator'], architecture['route_fraction_denominator']
    A = max(input_dim, L, (n*L*input_dim+den-1)//den)
    assert A <= L*input_dim
    # Cover every input coordinate at initialization and every hidden sum.
    edges = {(i % L, i % input_dim) for i in range(max(input_dim, L))}
    seed = int.from_bytes(hashlib.sha256(f'{architecture["topology_seed"]}:{input_dim}:{L}'.encode()).digest()[:8], 'little')
    rng = random.Random(seed)
    remaining = sorted(set((j, i) for j in range(L) for i in range(input_dim))-edges)
    rng.shuffle(remaining)
    edges.update(remaining[:A-len(edges)])
    assert len(edges) == A
    assert {j for j, _ in edges} == set(range(L))
    assert {i for _, i in edges} == set(range(input_dim))
    return [sorted(i for j, i in edges if j == branch) for branch in range(L)]


def specification(input_dim, architecture):
    G = architecture['G']
    mode = architecture.get('input_mode', 'full_affine')
    if mode not in ('full_affine', 'coordinate'):
        raise ValueError(mode)
    children = []
    for coordinates in layout(input_dim, architecture):
        routes = []
        for coordinate in coordinates:
            if mode == 'coordinate':
                routes.append(dict(input=coordinate))
            else:
                weights = [float(i == coordinate) for i in range(input_dim)]
                routes.append(dict(affine=weights, bias=0.0, learnable=True))
        children.append(dict(operation='sum', children=routes, phi=[G]*len(routes)))
    return dict(operation='sum', children=children, phi=[G]*len(children))


def parameter_count(input_dim, architecture):
    A = sum(map(len, layout(input_dim, architecture)))
    L, G = architecture['hidden_sums'], architecture['G']
    mode = architecture.get('input_mode', 'full_affine')
    if mode not in ('full_affine', 'coordinate'):
        raise ValueError(mode)
    return (0 if mode == 'coordinate' else A*(input_dim+1))+(A+L)*(G+2)
