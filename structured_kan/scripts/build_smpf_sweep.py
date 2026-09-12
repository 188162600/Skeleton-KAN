"""Freeze the dimension-adaptive 18-candidate ablation; metadata arithmetic only."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import importlib.util

PROJECT=Path(__file__).resolve().parents[1]
module_spec=importlib.util.spec_from_file_location('smpf_metadata',PROJECT/'structure_builder/smpf.py')
smpf=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(smpf)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==payload,f'Refusing to replace frozen config: {path}'
    else:path.write_text(payload)


def main():
    architectures=smpf.architecture_sweep()
    template=PROJECT/'data/catalogues/smpf_structure/smpf_L248_fraction123_G10_30.json'
    write(template,dict(schema='structured-kan.fixed-structure-ablation.v1',method=smpf.METHOD,
        reference='https://ojs.aaai.org/index.php/AAAI/article/view/25816',
        interpretation='SMPF-inspired two-stage sums; kernels and full learned affine input routes replace native primitives and coordinate inputs; no genetic search',
        formula='sum_l psi_l(sum_j phi_lj(a_lj dot x + b_lj))',
        route_budget='A=max(d,L,ceil(p*L*d)); p in {1/3,2/3,1}; fixed dimension-only topology seed',
        sharing='private affine and private phi',output_phi=False,
        heldout_expression_information_used=False,architectures=architectures))
    tref=dict(path=template.relative_to(PROJECT).as_posix(),sha256=hashlib.sha256(template.read_bytes()).hexdigest())
    base=json.loads((PROJECT/'data/train_config/mlp18_domain80.json').read_text())
    base.update(schema='structured-kan.smpf-structure-domain80.v1',method=smpf.METHOD,
        architectures=architectures,template=tref,report_floor=0,
        initialization='coordinate-covering private affine routes; train-only calibrated frozen grids; seed-dependent0.001 perturbations',
        width_rule='0.2*G**(2/3)',initialization_noise=.001,output_phi=False)
    for shard in range(3):
        c=dict(base,equation_partition=dict(policy='within-domain-round-robin',count=3,index=shard),
            output=f'data/results/smpf_structure_transfer80_shard{shard}')
        write(PROJECT/f'data/train_config/smpf_structure_domain80_shard{shard}.json',c)
    pde=json.loads((PROJECT/'data/train_config/ted_pde10.json').read_text())
    for key in ('catalogue','construction','G_rule'):pde.pop(key,None)
    pde.update(method=smpf.METHOD,architectures=architectures,template=tref,report_floor=0,
        output_phi=False,G_rule='uniform per candidate, G10 or G30',
        output='data/results/smpf_structure_pde10')
    write(PROJECT/'data/train_config/smpf_structure_pde10.json',pde)
    rows=json.loads((PROJECT/base['subset']['path']).read_text())
    problems=json.loads((PROJECT/pde['problems']['path']).read_text())['problems']
    cohorts={'transfer80':Counter(r['variables'] for r in rows),
             'pde10':Counter(len(p['coordinates']) for p in problems)}
    audit={}
    for cohort,distribution in cohorts.items():
        values={}
        for a in architectures:
            values[a['id']]=sum(n*smpf.parameter_count(d,a) for d,n in distribution.items())/sum(distribution.values())
        audit[cohort]=dict(dimension_distribution=dict(distribution),by_config=values,
            mean_all18=sum(values.values())/18,
            mean_G10=sum(values[a['id']] for a in architectures if a['G']==10)/9,
            mean_G30=sum(values[a['id']] for a in architectures if a['G']==30)/9)
    audit['counts_by_dimension']={str(d):{a['id']:smpf.parameter_count(d,a) for a in architectures} for d in range(1,11)}
    audit['expected_fits']=dict(transfer=80*18*10,pde=10*18*10,total=90*18*10)
    dest=PROJECT/'data/results/smpf_ablation_launch/parameter_audit.json';write(dest,audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
