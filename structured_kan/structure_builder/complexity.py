"""Compatibility wrapper: source complexity analysis, then Gaussian realization.

``mean_complexity`` is optional analyser metadata at each phi slot, not a
requirement of skeleton synthesis or realization. For the established dynamic-G
experiments, a separate parameter-budgeting policy consumes this annotation
and supplies ``G`` and ``phi_basis_by_label``. Uniform-G realizations need not
use the annotation. Both steps remain packaged here for frozen-format
compatibility; neither their heuristic connection nor either step provides an
approximation bound. Complexity is not itself a basis count.
"""
import copy
import hashlib
import json
from pathlib import Path
import statistics
from collections import defaultdict

from . import incidence as structural
from . import matching as tm
from .complexity_primitives import annotated, score


def allocate(catalogue_path, source_path):
    catalogue_path, source_path=Path(catalogue_path),Path(source_path)
    group=json.loads(catalogue_path.read_text());payload=source_path.read_bytes()
    sources=json.loads(payload)
    assert hashlib.sha256(payload).hexdigest()==group['training_sha256']
    assert len(sources)==group['training_equations'] and len(sources)>0
    assert all(r['source_corpus']!=group['heldout_domain'] for r in sources)
    assert group['builder_encoding']=='native-finite-topology-v1'
    native=group['builders'];assert len(native)==18
    def bn(t):return tm.Node(t['operator'],int(t['raw_arity']),tuple(bn(c) for c in t['children']))
    def tn(t):return tm.Node(t['op'],len(t['raw']),tuple(tn(c) for c in t['children']))
    def flat(n):
        out=[]
        def visit(n,path='root'):
            out.append((path,n))
            for i,c in enumerate(n['children']):visit(c,path+'.'+str(i))
        visit(n);return out
    bnodes=[bn(b['synthetic_operator_tree']) for b in native]
    assigned=defaultdict(list);checks=[];means=[];vocabulary={}
    for row in sources:
        spec=row['operator_variable_spec']
        for edge in [s.get('edge',{}) for n in spec['nodes'] for kind in ('raw_sources','state_sources') for s in n.get(kind,[])]+spec.get('output_edges',[]):
            value,label=score(edge);text=edge.get('symbolic_expression',edge.get('primitive','x'))
            vocabulary[text]=dict(complexity=value,family=label)
        root,output,labels=annotated(spec,structural)
        mean=statistics.mean(q for q,_ in labels);means.append(mean)
        target=tn(root);choices=[]
        for i,b in enumerate(bnodes):
            fit=tm.match(target,b,0)
            edges=sum(1+n.raw for n,_,_ in tm.flatten(b))
            choices.append(((fit.total,fit.mismatch,edges,native[i]['builder']),i,fit))
        _,index,fit=min(choices,key=lambda item:item[0])
        tf,bf=flat(root),flat(native[index]['synthetic_operator_tree']);maps={}
        for ti,bi in fit.node_pairs:
            path,b=bf[bi];_,t=tf[ti]
            for j,(_,q) in enumerate(sorted(t['raw'],key=lambda r:(-r[1][0],r[0]))[:b['raw_arity']]):
                maps[path+':raw:'+str(j)]=q[0]
            if path!='root':maps[path+':incoming']=t['incoming'][0]
        maps['output']=output[0]
        assigned[index].append(dict(equation=row['equation'],source_mean=mean,maps=maps,
                                    mismatch=fit.mismatch,unused=fit.unused))
        checks.append(dict(equation=row['equation'],builder=native[index]['builder'],mismatch=fit.mismatch,unused=fit.unused))
    result=copy.deepcopy(group)
    for index,builder in enumerate(result['builders']):
        ordered=[]
        def visit(tree,path='root'):
            for j,child in enumerate(tree['children']):visit(child,path+'.'+str(j))
            ordered.extend(path+':raw:'+str(j) for j in range(tree['raw_arity']))
            ordered.extend(path+'.'+str(j)+':incoming' for j in range(len(tree['children'])))
        visit(builder['synthetic_operator_tree']);ordered.append('output')
        allocation={};records=[]
        for label,key in enumerate(ordered):
            vals=[r['maps'].get(key,r['source_mean']) for r in assigned[index]]
            average=statistics.mean(vals) if vals else statistics.mean(means)
            # Analysis ends at `average`; the following quantization is the
            # current Gaussian realization policy, not a predicted basis count.
            budget=min((3,9,15),key=lambda g:(abs(g-average),g));name='phi_'+str(label)
            allocation[name]=budget
            records.append(dict(phi=name,route=key,mean_complexity=average,G=budget,
                matched=sum(key in r['maps'] for r in assigned[index]),assigned_sources=len(vals)))
        builder['phi_basis_by_label']=allocation
        builder['output_phi']=True
        builder['phi_complexity_allocation']=dict(source_only=True,source_sha256=group['training_sha256'],maps=records)
    result['complexity_allocation']=dict(rule='source-aligned average, nearest 3/9/15, lower tie',
        unmatched='source equation mean; empty builder uses source-pool mean',source_rows=len(sources),heldout_rows_read=0,
        source_assignments=checks,vocabulary=vocabulary,
        source_mean_mismatch=statistics.mean(r['mismatch'] for r in checks),
        source_mean_unused=statistics.mean(r['unused'] for r in checks),
        base_catalogue_sha256=hashlib.sha256(catalogue_path.read_bytes()).hexdigest())
    return result
