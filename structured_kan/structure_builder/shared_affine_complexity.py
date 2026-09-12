"""Source-only v2 private-map complexity, retaining original core/reserve rules."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict
from . import matching as tm,incidence
from .complexity_primitives import annotated,score


def allocate(catalogue_path,source_path):
    catalogue_path,source_path=Path(catalogue_path),Path(source_path)
    group=json.loads(catalogue_path.read_text());source=source_path.read_bytes();rows=json.loads(source)
    assert hashlib.sha256(source).hexdigest()==group['training_sha256']
    assert group['builder_encoding']=='shared-reserve-constructor-records-v1'
    assert len(rows)==group['training_equations'] and len(rows) in (225,300)
    assert all(r['source_corpus']!=group['heldout_domain'] for r in rows)
    assert len(group['builders'])==18
    def flat(t):
        result=[]
        def walk(n,p='root'):
            result.append((p,n))
            for i,c in enumerate(n['children']):walk(c,p+'.'+str(i))
        walk(t);return result
    def target(n):return tm.Node(n['op'],len(n['raw']),tuple(target(c) for c in n['children']))
    assigned=defaultdict(list);checks=[];means=[];vocabulary={}
    for row in rows:
        spec=row['operator_variable_spec'];d=len(spec['variables'])
        ann,output,labels=annotated(spec,incidence);tn=target(ann)
        avg=statistics.mean(q for q,_ in labels);means.append(avg);choices=[]
        for e in [s.get('edge',{}) for n in spec['nodes'] for kind in ('raw_sources','state_sources') for s in n.get(kind,[])]+spec.get('output_edges',[]):
            g,family=score(e);vocabulary[e.get('symbolic_expression',e.get('primitive','x'))]=dict(complexity=g,family=family)
        for i,b in enumerate(group['builders']):
            pool=b['raw_incidence_rule']['port_capacity_by_node']['__shared_pool__']
            bn=tm.builder_node(b['synthetic_operator_tree'],d,pool['core_capacity_by_node'])
            reserve=int(pool['reserve_capacity']);fit=tm.match(tn,bn,reserve)
            edges=sum(1+n.raw for n,_,_ in tm.flatten(bn))+len(tm.flatten(bn))*reserve
            choices.append(((fit.total,fit.mismatch,edges,b['builder']),i,bn,fit))
        _,index,bn,fit=min(choices);tf=flat(ann);bf=flat(group['builders'][index]['synthetic_operator_tree']);nf=tm.flatten(bn)
        maps={};overflow=[]
        for ti,bi in fit.node_pairs:
            path,_=bf[bi];_,t=tf[ti];raw=sorted(t['raw'],key=lambda v:(-v[1][0],v[0]));capacity=nf[bi][0].raw
            for j,(_,q) in enumerate(raw[:capacity]):maps[f'{path}:raw:{j}']=q[0]
            for j,(_,q) in enumerate(raw[capacity:]):overflow.append((path,j,q[0]))
            if path!='root':maps[path+':incoming']=t['incoming'][0]
        for lane,(path,j,q) in enumerate(sorted(overflow)[:fit.reserve_used]):maps[f'{path}:reserve:{lane}']=q
        maps['output']=output[0]
        assigned[index].append(dict(case_id=row['case_id'],mean=avg,maps=maps))
        checks.append(dict(case_id=row['case_id'],builder=group['builders'][index]['builder'],
            mismatch=fit.mismatch,unused=fit.unused,reserve_used=fit.reserve_used))
    result=copy.deepcopy(group);global_mean=statistics.mean(means)
    for index,b in enumerate(result['builders']):
        pool=b['raw_incidence_rule']['port_capacity_by_node']['__shared_pool__'];keys=[]
        assert b['raw_incidence_rule']['role_incidence_prior']['mode']=='independent-prefix'
        for path,n in flat(b['synthetic_operator_tree']):
            keys += [f'{path}:raw:{j}' for j in range(pool['core_capacity_by_node'][path])]
            keys += [f'{path}:reserve:{j}' for j in range(pool['reserve_capacity'])]
            if path!='root':keys.append(path+':incoming')
        keys.append('output');records=[];plan={}
        for key in keys:
            avg=statistics.mean(r['maps'].get(key,r['mean']) for r in assigned[index]) if assigned[index] else global_mean
            g=min((3,9,15),key=lambda n:(abs(n-avg),n));plan[key]=g
            records.append(dict(route=key,mean_complexity=avg,G=g,assigned_sources=len(assigned[index]),
                                matched=sum(key in r['maps'] for r in assigned[index])))
        b['private_phi_G_by_route']=plan
        b['phi_complexity_allocation']=dict(source_only=True,maps=records)
    result['complexity_allocation']=dict(source_rows=len(rows),heldout_rows_read=0,source_only=True,
        source_assignments=checks,vocabulary=vocabulary,stable_reserve_keys=True,
        rule='Source M+U assignment; private occurrence means; nearest3/9/15 lower tie; no cross-edge phi pooling',
        unmatched='source equation mean; empty builder uses source-fold mean',
        base_catalogue_sha256=hashlib.sha256(catalogue_path.read_bytes()).hexdigest())
    return result
