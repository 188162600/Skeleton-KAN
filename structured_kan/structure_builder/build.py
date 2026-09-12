"""Construct catalogues from source equations, not from existing builders.

Native AST clustering/generalization precedes finite representative selection
and phi projection. Held-out equations are never passed to this entry point.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from .cardinality import next_count,select_representatives
from .io import sha,write
from .terms import term_from_spec


def _distance_pair(job):
    from .native_trees import ordered_distance
    i,j,a,b=job
    return i,j,ordered_distance(a,b)


def _prepare_ted(trees,out,workers):
    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    distances=np.zeros((len(trees),len(trees)),dtype=np.float64)
    jobs=((i,j,trees[i],trees[j]) for i in range(len(trees)) for j in range(i+1,len(trees)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done,(i,j,value) in enumerate(pool.map(_distance_pair,jobs,chunksize=32),1):
            distances[i,j]=distances[j,i]=value
            if done%500==0:
                write(out/'progress.json',dict(phase='original_AST_distances',pairs=done,total=len(trees)*(len(trees)-1)//2))
    hierarchy=linkage(squareform(distances),method='average') if len(trees)>1 else np.empty((0,4))
    np.savez_compressed(out/'original_ted.npz',distances=distances,linkage=hierarchy)


def build_catalogue(source,output,*,method,k=18,heldout_domain=None,workers=8,max_count_builds=14,
                    pair_timeout_seconds=30.0):
    """Build one complete group JSON; skip-pair ACUOS2 is explicitly approximate.

    method is ks_ies, ted, fgw_sum or acuos2 (v2 is a compatibility alias).
    No existing catalogue is an input.
    Frozen operator-variable specifications supply the common phi projection;
    ordinary declared-input expression ASTs supply every native method.
    """
    import numpy as np
    from .native_trees import original_tree,groups_at
    if method in ('ks_ies','v2'):
        from .proposed import build_proposed
        return build_proposed(source,output,k=k,heldout_domain=heldout_domain)
    if method not in ('ted','fgw_sum','acuos2'):
        raise ValueError('Supported constructors: ks_ies, ted, fgw_sum, acuos2 (v2 alias)')
    source,output=Path(source),Path(output)
    rows=json.loads(source.read_text(encoding='utf-8'))
    if not rows or len({(r['source_corpus'],r['case_id']) for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate source equations')
    if heldout_domain is not None and any(r['source_corpus']==heldout_domain for r in rows):
        raise ValueError('Held-out domain leaked into construction inputs')
    if not 1<=k<=len(rows) or not 1<=workers<=64:
        raise ValueError('Invalid K or worker count')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh build directory; completed native outputs are not overwritten')
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    write(output/'source.json',rows)
    write(output/'progress.json',dict(phase='parsing',source_equations=len(rows),method=method))
    trees=[original_tree(row) for row in rows]
    terms=[term_from_spec(row['operator_variable_spec']) for row in rows]
    write(output/'original_asts.json',[t.payload() for t in trees])
    if method!='fgw_sum':
        _prepare_ted(trees,output,workers)
    vendor=Path(__file__).resolve().parent/'vendor'
    results=[]
    upper=len(rows)
    for attempt in range(max_count_builds):
        count=next_count(results,k,upper=upper)
        if count is None:break
        folder=output/f'native_{count:03d}';folder.mkdir()
        write(output/'progress.json',dict(phase='native_construction',method=method,native_k=count,attempt=attempt+1))
        iteration_start=time.monotonic()
        if method=='fgw_sum':
            from .fgw_pot import native_fgw_pot
            groups,reps,details=native_fgw_pot(trees,count,vendor/'fgw_published_3d2128a',folder)
        else:
            groups,distances=groups_at(output,count)
            if method=='ted':
                reps=[min(g,key=lambda i:(float(distances[i,g].sum()),trees[i].signature(),i)) for g in groups]
                details=dict(adapter='ordered unit-cost APTED + average linkage + finite cluster medoid')
                write(folder/'native.json',dict(groups=groups,medoid_indices=reps))
            else:
                from .acuos2 import native_catalogue
                reps,details=native_catalogue(trees,groups,vendor/'acuos2_official_d9214e03',folder,
                    theory='ACU',pair_timeout_seconds=pair_timeout_seconds,timeout_policy='skip_merge')
        distinct={}
        for cluster,index in enumerate(reps):
            signature=terms[index].signature()
            distinct.setdefault(signature,dict(source_index=int(index),cluster=cluster,
                cluster_size=len(groups[cluster]),signature=signature,equation=rows[index]['equation']))
        result=dict(status='succeeded',native_k=count,final_distinct_count=len(distinct),
                    representatives=list(distinct.values()),seconds=time.monotonic()-iteration_start,**details)
        results.append(result);write(folder/'result.json',result)
        write(output/'counts.json',results)
    chosen,selected,added,exact=select_representatives(results,k)
    if len(selected)!=k:
        write(output/'progress.json',dict(phase='incomplete',builders_found=len(selected),requested=k,
              reason='Native count budget supplied too few distinct finite builders; no duplicate padding'))
        raise RuntimeError(f'Found {len(selected)} distinct builders, requested {k}')
    builders=[]
    for index,representative in enumerate(selected,1):
        term=terms[representative['source_index']]
        builders.append(dict(builder=f'IAPB{index:02d}',synthetic_operator_tree=json.loads(term.signature()),
            arithmetic_nodes=term.nodes,operator_pattern=term.notation('X'),
            source_representative=representative,materialization='unshared finite source topology'))
    name=f'{method}_{heldout_domain}_k{k}'
    group=dict(schema='structured-kan.catalogue-group.v1',name=name,method=method,
        heldout_domain=heldout_domain,builder_count=k,training_equations=len(rows),
        training_sha256=sha(source),source_only=True,source_domains=sorted({r['source_corpus'] for r in rows}),
        builder_encoding='native-finite-topology-v1',builders=builders,
        construction=dict(native_input='original declared-input expression AST',
            phi_projection='after native representatives, using frozen executable-edge source specifications',
            existing_catalogue_used=False,heldout_data_supplied=False,
            native_counts=[r['native_k'] for r in results],shortfall_rule='K + missing distinct phi builders',
            exact_native_hit=exact,added_from_larger_results=added,
            acuos2_policy='skip timed-out pair and retain previous verified pattern; approximate n-way catalogue' if method=='acuos2' else None,
            pair_timeout_seconds=pair_timeout_seconds if method=='acuos2' else None,
            seconds=time.monotonic()-start,base_native_k=chosen['native_k']))
    write(output/'catalogue.json',group)
    write(output/'progress.json',dict(phase='completed',catalogue_sha256=sha(output/'catalogue.json'),
        builders=k,seconds=time.monotonic()-start))
    return group
