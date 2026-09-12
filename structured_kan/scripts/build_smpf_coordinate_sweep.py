"""Freeze coordinate-input successors; preserve every original frozen artifact."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('smpf_metadata',PROJECT/'structure_builder/smpf.py')
smpf=importlib.util.module_from_spec(spec);spec.loader.exec_module(smpf)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==payload,f'Frozen artifact differs: {path}'
    else:path.write_text(payload)


def main():
    architectures=smpf.architecture_sweep(input_mode='coordinate')
    old_arch=smpf.architecture_sweep()
    assert [{k:v for k,v in a.items() if k!='input_mode'} for a in architectures]==old_arch
    template=PROJECT/'data/catalogues/smpf_structure/smpf_coordinate_L248_fraction123_G10_30.json'
    write(template,dict(schema='structured-kan.fixed-structure-ablation.v2',method=smpf.METHOD,
        reference='https://ojs.aaai.org/index.php/AAAI/article/view/25816',input_mode='coordinate',
        interpretation='SMPF-inspired coordinate-to-hidden interaction groups with kernel unary maps; fixed sweep, no genetic search',
        formula='sum_l psi_l(sum_{j in S_l} phi_lj(x_j))',
        connectivity='Same source-independent dimension-only masks as original sweep; fixed coordinate leaves, no learned cross-variable affine routes',
        sharing='private phi; coordinate leaves have no parameters',output_phi=False,
        heldout_expression_information_used=False,architectures=architectures))
    tref=dict(path=template.relative_to(PROJECT).as_posix(),sha256=hashlib.sha256(template.read_bytes()).hexdigest())
    changes={}
    for suffix in ('domain80_shard0','domain80_shard1','domain80_shard2','pde10'):
        old_path=PROJECT/f'data/train_config/smpf_structure_{suffix}.json'
        old=json.loads(old_path.read_text());new=dict(old,architectures=architectures,template=tref,input_mode='coordinate',
            output=old['output'].replace('smpf_structure_','smpf_coordinate_'),
            initialization='fixed coordinate leaves; train-only calibrated frozen grids; seeded0.001 perturbations of unary-map parameters only')
        new_path=PROJECT/f'data/train_config/smpf_coordinate_{suffix}.json'
        write(new_path,new)
        keys={k for k in set(old)|set(new) if old.get(k)!=new.get(k)}
        assert keys=={'architectures','template','input_mode','output','initialization'}
        changes[suffix]=dict(old_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
            new_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),changed_keys=sorted(keys))
    rows=json.loads((PROJECT/new['source']['path']).read_text())
    transfer=json.loads((PROJECT/'data/train_config/smpf_coordinate_domain80_shard0.json').read_text())
    selected=json.loads((PROJECT/transfer['subset']['path']).read_text())
    problems=json.loads((PROJECT/new['problems']['path']).read_text())['problems']
    assert len(rows)==300 and len(selected)==80 and len(problems)==10
    stats={}
    for name,distribution in {'transfer80':Counter(r['variables'] for r in selected),
                             'pde10':Counter(len(r['coordinates']) for r in problems)}.items():
        counts={a['id']:sum(n*smpf.parameter_count(d,a) for d,n in distribution.items())/sum(distribution.values()) for a in architectures}
        stats[name]=dict(dimension_distribution=dict(distribution),by_config=counts,
            mean_all18=sum(counts.values())/18,
            mean_G10=sum(counts[a['id']] for a in architectures if a['G']==10)/9,
            mean_G30=sum(counts[a['id']] for a in architectures if a['G']==30)/9)
    audit=dict(changes=changes,parameter_counts=stats,expected_fits=dict(transfer80=14400,pde10=1800),
        unchanged='L,p,G,coordinate masks,seeds,sampling,normalization,optimizer,budget,workers,selection')
    write(PROJECT/'data/results/smpf_coordinate_relaunch/configuration_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
