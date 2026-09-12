"""Translate a frozen finite topology into explicit model children and maps.

Unlike synthesis, this operation does not inspect any source/target expression.
All Gaussian budgets and whether to add an output map are explicit arguments.
Shared-reserve records must use a separate adapter; they are rejected here.
"""
import copy


def finite_topology_spec(tree,input_dim,*,G=3,per_map_G=None,output_G=None,off_diagonal=0.01):
    """Materialize full affine raw routes; labels follow historical postorder.

    G is a fallback per-map budget. per_map_G, when supplied, must specify
    exactly every incoming phi_N (and the output phi_N if output_G is set).
    No output phi is inserted when output_G is None.
    """
    if type(input_dim) is not int or input_dim<1:raise ValueError('Invalid input dimension')
    if not 0<=off_diagonal<1:raise ValueError('off_diagonal must be in [0,1)')
    routes=0;maps=0;labels=[]

    def budget(default):
        nonlocal maps
        name='phi_'+str(maps);maps+=1;labels.append(name)
        value=per_map_G[name] if per_map_G is not None else default
        if type(value) is not int or value<2:raise ValueError('Each G must be an integer >=2')
        return value

    def visit(node):
        nonlocal routes
        if not isinstance(node,dict) or 'raw_arity' not in node:
            raise ValueError('Finite topology must provide exact raw_arity at every node')
        allowed={'operator','raw_arity','raw_attachment','children'}
        if set(node)-allowed:raise ValueError('Non-finite/shared capacity metadata requires its own adapter')
        raw=node['raw_arity']
        if type(raw) is not int or raw<0:raise ValueError('Invalid raw arity')
        children=[visit(child) for child in node.get('children',[])]
        raw_children=[]
        for _ in range(raw):
            weights=[off_diagonal]*input_dim;weights[routes%input_dim]=1.;routes+=1
            raw_children.append(dict(affine=weights,bias=0.,learnable=True))
        children=raw_children+children
        if not children:raise ValueError('An interaction must have at least one child')
        operator=node['operator']
        if operator not in ('+','*'):raise ValueError('Finite topology operator must be + or *')
        return dict(operation='sum' if operator=='+' else 'prod',children=children,
                    phi=[budget(G) for _ in children])

    specification=visit(copy.deepcopy(tree))
    if output_G is not None:
        specification=dict(operation='sum',children=[specification],phi=[budget(output_G)])
    if per_map_G is not None and set(per_map_G)!=set(labels):
        raise ValueError('Budget labels do not match the realized phi maps exactly')
    return specification
