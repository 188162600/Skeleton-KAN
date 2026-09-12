"""v2 core/reserve realization with shared affine roles and PRIVATE phi maps.

Only source-frozen independent-prefix v2 is accepted. Route identities follow
the original materializer; reserve IDs are stable even when d caps the core.
There are no phi aliases, learned gates outside phi, or extra affine outputs.
"""


def shared_affine_layout(builder,input_dim):
    if type(input_dim) is not int or input_dim<1:raise ValueError('Invalid input dimension')
    rule=builder['raw_incidence_rule'];pool=rule['port_capacity_by_node']['__shared_pool__']
    if rule.get('role_incidence_prior',{}).get('mode')!='independent-prefix':
        raise ValueError('This adapter only supports source-frozen v2 independent-prefix rules')
    if pool.get('sparse_reserve_capacity',0):raise ValueError('Sparse reserve requires a separate adapter')
    core=pool['core_capacity_by_node'];reserve=int(pool['reserve_capacity'])
    if reserve<0 or any(int(c)<0 for c in core.values()):raise ValueError('Negative capacity')
    capacities={p:min(input_dim,int(c)) for p,c in core.items()}
    reserve_base=max(capacities.values(),default=0)
    state_priors=builder['edge_route_rule'].get('state_route_prior',{})
    ordered=[];nodes=[]
    def visit(t,path='root'):
        if t['operator'] not in ('+','*'):raise ValueError('Only v2 sum_phi/prod_phi nodes are supported')
        children=[visit(c,path+'.'+str(i)) for i,c in enumerate(t['children'])]
        neutral=0. if t['operator']=='+' else 1.
        capacity=capacities[path]
        active=min(capacity,int(rule['activation_prior_by_node'][path]['initial_active_lanes']))
        raw=[dict(role=j,key=f'{path}:raw:{j}',initial_a=float(j<active),
                  initial_b=0. if j<active else neutral) for j in range(capacity)]
        raw += [dict(role=reserve_base+j,key=f'{path}:reserve:{j}',initial_a=0.,initial_b=neutral)
                for j in range(reserve)]
        state=[]
        for i,c in enumerate(children):
            cp=path+'.'+str(i);prior=state_priors.get(path+'->'+cp,{})
            used=int(prior.get('used_equations',1))>=int(prior.get('unused_equations',0))
            state.append(dict(key=cp+':incoming',initial_a=float(used),initial_b=0. if used else neutral))
        if not raw and not children:raise ValueError('Empty materialized node')
        incoming=raw+state
        for edge in incoming:edge['phi']='phi_'+str(len(ordered));ordered.append(edge)
        node=dict(operator=t['operator'],path=path,raw=raw,children=children,state=state)
        nodes.append(node);return node
    root=visit(builder['synthetic_operator_tree'])
    output=dict(key='output',initial_a=1.,initial_b=0.,phi='phi_'+str(len(ordered)))
    ordered.append(output)
    roles=sorted({e['role'] for e in ordered if 'role' in e})
    return dict(root=root,output=output,edges=ordered,nodes=nodes,roles=roles,
                reserve_base=reserve_base,reserve_capacity=reserve)


def shared_affine_private_phi_spec(builder,input_dim,*,G=3,off_diagonal=0.,share_affine=True,share_phi=False):
    if type(share_affine) is not bool:raise ValueError('share_affine must be boolean')
    if type(share_phi) is not bool:raise ValueError('share_phi must be boolean')
    if share_phi and share_affine:raise ValueError('This ablation requires private affine routes')
    layout=shared_affine_layout(builder,input_dim)
    plan=builder.get('private_phi_G_by_route')
    budgets={e['phi']:int(plan[e['key']]) if plan is not None else G for e in layout['edges']}
    if any(type(g) is not int or g<2 for g in budgets.values()):raise ValueError('Invalid G')
    groups={};group_by_label={}
    for e in layout['edges']:
        # Preserve source role, occurrence-specific G and active/neutral priors.
        # State/output edges have no raw source role and remain private.
        key=(e['role'],budgets[e['phi']],e['initial_a'],e['initial_b']) if share_phi and 'role' in e else ('private',e['phi'])
        if key not in groups:groups[key]='phi_group_'+str(len(groups))
        group_by_label[e['phi']]=groups[key]
    def edge(e):
        item=dict(G=budgets[e['phi']],initial_a=e['initial_a'],initial_b=e['initial_b'])
        if share_phi and 'role' in e:item['share']=group_by_label[e['phi']]
        return item
    def visit(n):
        raw=[]
        for e in n['raw']:
            weights=[off_diagonal]*input_dim;weights[e['role']%input_dim]=1.
            route=dict(affine=weights,bias=0.,learnable=True)
            if share_affine:route['share']='affine_role_'+str(e['role'])
            raw.append(route)
        return dict(operation='sum' if n['operator']=='+' else 'prod',
            children=raw+[visit(c) for c in n['children']],phi=[edge(e) for e in n['raw']+n['state']])
    spec=dict(operation='sum',children=[visit(layout['root'])],phi=[edge(layout['output'])])
    occurrences=sum(len(n['raw']) for n in layout['nodes'])
    affine_modules=len(layout['roles']) if share_affine else occurrences
    unique_budgets={group_by_label[label]:g for label,g in budgets.items()}
    meta=dict(shared_affine_roles=len(layout['roles']) if share_affine else 0,
        source_affine_roles=len(layout['roles']),affine_route_modules=affine_modules,raw_routes=occurrences,
        interaction_nodes=len(layout['nodes']),scalar_edges=len(layout['edges']),
        phi_basis_by_label=budgets,phi_route_by_label={e['phi']:e['key'] for e in layout['edges']},
        phi_modules=len(unique_budgets),phi_reused_occurrences=len(budgets)-len(unique_budgets),
        phi_group_by_label=group_by_label,
        expected_parameters=affine_modules*(input_dim+1)+sum(g+2 for g in unique_budgets.values()),
        fully_separate_parameters=occurrences*(input_dim+1)+sum(g+2 for g in budgets.values()),
        sharing=('private affine routes; shared whole phi within source-role/G/initial-prior groups; private state/output phi'
                 if share_phi else ('shared affine routes' if share_affine else 'private affine routes')+'; private phi per edge including reserve, state and output'))
    return spec,meta


def private_affine_shared_phi_spec(builder,input_dim,*,G=3,off_diagonal=0.):
    """Tie whole raw phi maps, never affine routes or incompatible priors.

    Source-only dynamic G is unchanged. All affine occurrences are independent;
    even routes feeding one shared map can learn different input projections.
    """
    return shared_affine_private_phi_spec(builder,input_dim,G=G,off_diagonal=off_diagonal,
                                         share_affine=False,share_phi=True)


def initialize_private_from_shared_(model,shared_spec,train_inputs,seed,std=.001):
    """Copy the paired shared run's initial values, without retaining aliases.

    State-dict loading copies each aliased source value into distinct destination
    parameters. The temporary tied model is never optimized or retained.
    """
    from .StructuredKANBuilder import StructuredKANBuilder
    from .initialization import perturb_trainable_
    parameter=next(model.parameters())
    paired=StructuredKANBuilder(train_inputs.shape[-1],dtype=parameter.dtype,
        device=parameter.device).build(shared_spec,train_inputs=train_inputs)
    source_digest=perturb_trainable_(paired,seed,std)
    model.load_state_dict(paired.state_dict(),strict=True)
    digest=perturb_trainable_(model,seed,0.)
    return digest,source_digest
