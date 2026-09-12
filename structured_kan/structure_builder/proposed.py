"""KS-inspired Interaction Envelope Synthesis (KS-IES; proposed).

The verified source-only envelope, incidence and shared-reserve constructor.
The implementation key remains ``v2`` for frozen-catalogue compatibility.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

from .io import sha,write
from .terms import Term


def term_from_payload(value):
    return Term(value['operator'],bool(value['raw_attachment']),
        tuple(term_from_payload(c) for c in value.get('children',[])),int(value.get('raw_arity',0)))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_proposed(source,output,*,k=18,heldout_domain=None):
    """Build KS-IES (proposed), preserving v2 objectives; prohibit held-out rows."""
    from . import synthesis
    source,output=Path(source).resolve(),Path(output).resolve()
    rows=json.loads(source.read_text())
    if not rows or len({(r['source_corpus'],r['case_id']) for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate source equations')
    if heldout_domain is not None and any(r['source_corpus']==heldout_domain for r in rows):
        raise ValueError('Held-out domain leaked into source construction')
    if not 1<=k<=len(rows):raise ValueError('Invalid K')
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();commands=[]
    project=Path(__file__).resolve().parents[1]
    # Legacy audit writers require an integer row index. It is bookkeeping,
    # never an input to synthesis or source-frequency weighting.
    rows=[dict(r,index=r.get('index',i)) for i,r in enumerate(rows)]
    native_source=output/'source.json'
    write(native_source,rows)

    def run(module,*args):
        command=[sys.executable,'-m','structured_kan.structure_builder.'+module,*map(str,args)]
        commands.append(command)
        write(output/'progress.json',dict(phase=module,source_only=True,commands=commands))
        with (output/(module+'.log')).open('w') as stream:
            process=subprocess.run(command,cwd=project.parent,stdout=stream,stderr=subprocess.STDOUT)
        if process.returncode:
            tail=(output/(module+'.log')).read_text(errors='replace')[-4000:]
            raise RuntimeError(f'{module} exited {process.returncode}:\n{tail}')

    stage1,incidence,stage2=output/'stage1',output/'incidence',output/'stage2'
    run('synthesis','--input',native_source,'--output',stage1,'--builders',k,'--dimension-weight',0,
        '--class-weighting','frequency','--structural-score','unused-capacity','--ignore-input-dimension',
        '--refinement-swaps',6,'--collapse-univariate-chains')
    frozen=read_jsonl(stage1/'builders.jsonl')
    if len(frozen)!=k:raise RuntimeError('Source synthesis did not produce the requested builder count')
    clusters=[dict(envelope=term_from_payload(r['synthetic_operator_tree']),dimension_values=()) for r in frozen]
    diagnostic=synthesis.evaluate_fixed_catalogue(rows,clusters,dimension_weight=0.,ignore_input_dimension=True,
        structural_score='unused-capacity',collapse_univariate_chains=True)
    with (stage1/'evaluation_assignments.jsonl').open('w') as stream:
        for row in diagnostic['assignments']:stream.write(json.dumps(row)+'\n')
    # Historical stage names accept an evaluation argument: it receives SOURCE
    # rows again, never the target fold, for both diagnostics and capacity rules.
    run('incidence','--external',native_source,'--feynman',native_source,'--catalogue',stage1,'--output',incidence,
        '--structural-score','unused-capacity','--raw-arity-policy','capacity-separated','--collapse-univariate-chains')
    run('shared_reserve','--external',native_source,'--feynman',native_source,'--catalogue',stage1,'--audit',incidence,
        '--output',stage2,'--structural-score','unused-capacity','--capacity-policy','compatible-shared-reserve',
        '--reserve-quantile',.95,'--global-capacity-quantile',.85,'--prior-population','assigned',
        '--role-incidence-prior','independent-prefix','--raw-arity-policy','capacity-separated','--collapse-univariate-chains')
    builders=read_jsonl(stage2/'builders.jsonl')
    if len(builders)!=k:raise RuntimeError('Shared-reserve stage changed the builder count')
    result=dict(schema='structured-kan.catalogue-group.v1',name=f'v2_{heldout_domain}_k{k}',method='v2',
        heldout_domain=heldout_domain,builder_count=k,training_equations=len(rows),training_sha256=sha(source),
        source_only=True,source_domains=sorted({r['source_corpus'] for r in rows}),
        builder_encoding='shared-reserve-constructor-records-v1',builders=builders,
        construction=dict(source_only=True,existing_catalogue_used=False,heldout_data_supplied=False,
            objective='verified v2 unused-capacity objective with frequency weighting and six source-only swaps',
            reserve_quantile=.95,global_capacity_quantile=.85,prior_population='assigned',
            role_incidence_prior='independent-prefix',native_counts=[k],commands=commands,seconds=time.monotonic()-start))
    write(output/'catalogue.json',result)
    write(output/'progress.json',dict(phase='completed',builders=k,seconds=time.monotonic()-start,
                                     catalogue_sha256=sha(output/'catalogue.json')))
    return result
