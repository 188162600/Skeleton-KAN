"""Durable Domain300 baseline training through the standalone model stack.

One JSON selects a method's four catalogues and the frozen evaluation cohort.
Each GPU subprocess fits ten independent seeds for exactly one equation and
builder, then exits, releasing its CUDA context and temporary compiler files.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

PROJECT=Path(__file__).resolve().parents[1]
SCHEMA='structured-kan.domain300-training.v1'
DOMAINS=('symbolic_search','physics','mathematics','biochemistry')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def inspect_config(filename):
    path=Path(filename)
    if not path.is_file():path=PROJECT/'data/train_config'/filename
    config=json.loads(path.read_text());assert config['schema']==SCHEMA
    assert config['seeds']==list(range(421,431)) and config['points_per_split']==10000
    assert config['outer_steps']==50 and config['width_rule']=='0.2*G**(2/3)'
    assert config['normalization']=='train-only-standardization'
    assert config['initialization_noise']==.001 and config['output_phi'] is True
    assert config['data_domains']==list(DOMAINS) and config['workers']==2
    subset=PROJECT/config['subset']['path'];assert sha(subset)==config['subset']['sha256']
    parent=PROJECT/config['parent']['path'];assert sha(parent)==config['parent']['sha256']
    rows=json.loads(subset.read_text());parents=json.loads(parent.read_text())
    assert len(rows)==80 and len(parents)==300
    parent_rows={r['case_id']:r for r in parents}
    assert len(parent_rows)==300 and len({r['case_id'] for r in rows})==80
    for row in rows:
        original=parent_rows[row['case_id']]
        for key in ('source_corpus','variables','analysis_expression','operator_variable_spec','quality_audit'):
            assert row[key]==original[key],(row['case_id'],key)
    for domain in DOMAINS:
        assert sum(r['source_corpus']==domain for r in rows)==20
        entry=config['catalogues'][domain];source=PROJECT/entry['source'];catalogue=PROJECT/entry['path']
        assert sha(source)==entry['source_sha256'] and sha(catalogue)==entry['sha256']
        group=json.loads(catalogue.read_text());sources=json.loads(source.read_text())
        assert group['heldout_domain']==domain and group['training_sha256']==entry['source_sha256']
        encoding=('shared-reserve-constructor-records-v1' if config['method']=='v2' else 'native-finite-topology-v1')
        if config['method']=='v2':
            assert config.get('materialization') in ('shared-affine-private-phi-v1','private-affine-private-phi-v1','private-affine-shared-phi-v1')
            if config['materialization']=='private-affine-shared-phi-v1':
                assert config['phi_sharing_policy']=='source-role-compatible-G-and-prior-v1'
            if config['materialization']=='private-affine-private-phi-v1':
                assert config.get('paired_initialization')=='shared-affine-private-phi-v1'
        assert group['method']==config['method'] and group['builder_encoding']==encoding
        assert len(group['builders'])==18 and len({r['builder'] for r in group['builders']})==18
        assert len(sources)==225 and all(r['source_corpus']!=domain for r in sources)
    equation_indices(config,rows)  # Validate any explicit disjoint partition.
    return path.resolve(),config,rows


def equation_indices(config,rows):
    partition=config.get('equation_partition')
    if partition is None:return list(range(len(rows)))
    assert partition['policy']=='within-domain-round-robin'
    index,count=partition['index'],partition['count']
    assert type(index) is int and type(count) is int and 0<=index<count and count>=1
    owned=[]
    for domain in DOMAINS:
        indices=[i for i,r in enumerate(rows) if r['source_corpus']==domain]
        owned.extend(indices[index::count])
    return owned


def fit_job(config_path,equation_index,builder_index):
    import torch
    from ..dataset.fresh_loader import load_seed_data
    from ..dataset.regression import regression_metrics
    from ..model import StructuredKANBuilder
    from ..model.topology_spec import finite_topology_spec
    from ..model.initialization import perturb_trainable_
    from ..optimizer.fit import fit_seed_batch
    path,config,rows=inspect_config(config_path)
    assert equation_index in equation_indices(config,rows),'Equation belongs to another shard'
    run=PROJECT/config['output'];row=rows[equation_index];domain=row['source_corpus']
    group_path=run/'catalogues'/(domain+'.json');group=json.loads(group_path.read_text())
    assert group['complexity_allocation']['base_catalogue_sha256']==config['catalogues'][domain]['sha256']
    builder=group['builders'][builder_index]
    job=run/'jobs'/row['audit_id'][:20]/builder['builder'];job.mkdir(parents=True,exist_ok=True)
    started=time.time()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU fallback')
    raw,receipt=load_seed_data(row,run/'sampling')
    standardized=[d.standardize() for d in raw]
    datasets=[d for d,_ in standardized];normalization=[n for _,n in standardized]
    materialization=None;paired_spec=None
    if config['method']=='v2':
        from ..model.shared_affine_spec import shared_affine_private_phi_spec
        if config['materialization']=='private-affine-shared-phi-v1':
            from ..model.shared_affine_spec import private_affine_shared_phi_spec
            spec,materialization=private_affine_shared_phi_spec(builder,row['variables'],off_diagonal=config['route_off_diagonal'])
        else:
            shared=config['materialization']=='shared-affine-private-phi-v1'
            spec,materialization=shared_affine_private_phi_spec(builder,row['variables'],off_diagonal=config['route_off_diagonal'],share_affine=shared)
            if not shared:
                paired_spec,_=shared_affine_private_phi_spec(builder,row['variables'],off_diagonal=config['route_off_diagonal'])
        budgets=materialization['phi_basis_by_label']
    else:
        budgets=builder['phi_basis_by_label']
        spec=finite_topology_spec(builder['synthetic_operator_tree'],row['variables'],per_map_G=budgets,
                                 output_G=budgets['phi_'+str(len(budgets)-1)],off_diagonal=config['route_off_diagonal'])
    assert set(budgets.values())<={3,9,15}
    models=[];initial=[];paired_initial=[]
    for data,seed in zip(datasets,config['seeds']):
        model=StructuredKANBuilder(row['variables'],device='cuda').build(spec,train_inputs=data.train.x)
        if paired_spec is None:
            initial.append(perturb_trainable_(model,seed,config['initialization_noise']))
        else:
            from ..model.shared_affine_spec import initialize_private_from_shared_
            digest,source_digest=initialize_private_from_shared_(model,paired_spec,data.train.x,seed,config['initialization_noise'])
            initial.append(digest);paired_initial.append(source_digest)
        models.append(model)
    assert len(set(initial))==10
    def counts(tree):
        return tree['raw_arity']+sum(counts(c)[0] for c in tree['children']),1+sum(counts(c)[1] for c in tree['children'])
    if materialization is not None:
        routes,nodes=materialization['raw_routes'],materialization['interaction_nodes']
        expected=materialization['expected_parameters']
    else:
        routes,nodes=counts(builder['synthetic_operator_tree'])
        expected=routes*(row['variables']+1)+sum(g+2 for g in budgets.values())
    parameters=sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    assert parameters==expected,(parameters,expected)
    frozen=[{n:t.detach().clone() for n,t in m.named_buffers()} for m in models]
    def progress(outer,loss,iterations,evaluations):
        save(job/'progress.json',dict(status='training',pid=os.getpid(),equation=row['equation'],
             builder=builder['builder'],outer=outer,outer_steps=50,seconds=time.time()-started,
             train_standardized_mse=loss.cpu().tolist(),inner_iterations=iterations,function_evaluations=evaluations))
    save(job/'progress.json',dict(status='compiling',pid=os.getpid(),equation=row['equation'],
         builder=builder['builder'],parameters=parameters,started=started))
    fit=fit_seed_batch(models,datasets,outer_steps=50,compile_mode=config['compile_mode'],progress=progress)
    assert all(torch.equal(t,dict(m.named_buffers())[n]) for m,buffers in zip(models,frozen) for n,t in buffers.items())
    records=[];states=[]
    for index,(model,data,norm) in enumerate(zip(models,datasets,normalization)):
        metrics={}
        with torch.no_grad():
            for split in ('train','validation','test'):
                prediction=model(getattr(data,split).x.cuda())*norm['y_std'].cuda()+norm['y_mean'].cuda()
                metrics[split]=regression_metrics(prediction,getattr(raw[index],split).y.cuda())
                assert all(math.isfinite(v) and v>=0 for v in metrics[split].values()),metrics
        record=dict(seed=config['seeds'][index],case_id=row['case_id'],equation=row['equation'],domain=domain,
            method=config['method'],builder=builder['builder'],parameters=parameters,metrics=metrics,
            initialization_sha256=initial[index],inner_iterations=fit.inner_iterations[index],
            function_evaluations=fit.function_evaluations[index])
        records.append(record);states.append({n:t.detach().cpu() for n,t in model.state_dict().items()})
    checkpoint=job/'models.pt';temporary=job/'models.tmp'
    torch.save(dict(schema='structured-kan.seed-batch-checkpoint.v1',structure=spec,states=states,
        seeds=config['seeds'],normalization=normalization,results=records,catalogue_sha256=sha(group_path),
        configuration_sha256=sha(path),data_receipt=receipt,materialization=materialization),temporary);temporary.replace(checkpoint)
    result=dict(status='complete',equation=row['equation'],case_id=row['case_id'],domain=domain,
        builder=builder['builder'],method=config['method'],parameters=parameters,raw_routes=routes,interaction_nodes=nodes,
        per_map_G=budgets,materialization=materialization,paired_initialization_sha256=paired_initial,seeds=records,seconds=time.time()-started,finished=time.time(),
        mean_log_validation_nmse=sum(math.log(max(r['metrics']['validation']['nmse'],1e-300)) for r in records)/10,
        configuration_sha256=sha(path),catalogue_sha256=sha(group_path),checkpoint_sha256=sha(checkpoint))
    save(job/'result.json',result);save(job/'progress.json',dict(status='complete',seconds=result['seconds'],finished=result['finished']))
    print(json.dumps(dict(equation=row['equation'],builder=builder['builder'],seconds=result['seconds'],parameters=parameters)),flush=True)


def summarize(run,rows):
    selected=[];batches=0
    for row in rows:
        records=[json.loads(p.read_text()) for p in (run/'jobs'/row['audit_id'][:20]).glob('*/result.json')]
        batches+=len(records)
        if len(records)==18:
            best=min(records,key=lambda r:(r['mean_log_validation_nmse'],r['builder']))
            def gm(metric):return math.exp(sum(math.log(max(r['metrics']['test'][metric],1e-10)) for r in best['seeds'])/10)
            selected.append(dict(equation=row['equation'],case_id=row['case_id'],domain=row['source_corpus'],
                builder=best['builder'],parameters=best['parameters'],test_gmse=gm('mse'),test_gnmse=gm('nmse')))
    result=dict(completed_batches=batches,completed_fits=10*batches,expected_fits=len(rows)*18*10,
        complete_equations=len(selected),selection='one builder per equation, mean log validation NMSE over ten seeds',
        report_floor=1e-10,selected=selected)
    if selected:
        for metric in ('test_gmse','test_gnmse'):
            result[metric]=math.exp(sum(math.log(r[metric]) for r in selected)/len(selected))
        result['mean_selected_parameters']=sum(r['parameters'] for r in selected)/len(selected)
    save(run/'summary.json',result)
    return result


def run_queue(config_path):
    path,config,rows=inspect_config(config_path);run=PROJECT/config['output'];run.mkdir(parents=True,exist_ok=True)
    owned=equation_indices(config,rows);owned_rows=[rows[i] for i in owned]
    lock=run/'queue.lock'
    # A stale lock is intentionally not stolen: inspect the recorded PID first.
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(status='preparing',pid=os.getpid(),method=config['method'],started=time.time(),configuration_sha256=sha(path),
               equation_partition=config.get('equation_partition'),expected_fits=len(owned)*18*10,expected_equations=len(owned))
    def update(**values):state.update(values,updated=time.time());save(run/'state.json',state)
    update()
    try:
        if config['method']=='v2':
            from ..structure_builder.shared_affine_complexity import allocate
        else:
            from ..structure_builder.complexity import allocate
        from ..dataset.fresh_sampling import generate_row
        from ..dataset.fresh_loader import load_seed_data
        for domain in DOMAINS:
            update(status='allocating_source_complexity',domain=domain)
            group_path=run/'catalogues'/(domain+'.json');entry=config['catalogues'][domain]
            if not group_path.exists():save(group_path,allocate(PROJECT/entry['path'],PROJECT/entry['source']))
            group=json.loads(group_path.read_text())
            assert group['complexity_allocation']['base_catalogue_sha256']==entry['sha256']
            indices=[i for i in owned if rows[i]['source_corpus']==domain]
            update(status='sampling',domain=domain,sampling_completed=0,sampling_expected=len(indices))
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(generate_row,(rows[i],run/'sampling',10000)) for i in indices]
                completed=0
                for future in as_completed(futures):
                    receipt=future.result();assert receipt['status']=='complete',receipt
                    completed+=1;update(sampling_completed=completed,last_sample=receipt['case_id'])
            # All thirty streams per equation must validate before this domain trains.
            for i in indices:load_seed_data(rows[i],run/'sampling')
            pending=[(i,b) for i in indices for b in range(18)
                     if not (run/'jobs'/rows[i]['audit_id'][:20]/group['builders'][b]['builder']/'result.json').exists()]
            active=[];update(status='training',pending_batches=len(pending),active=[])
            while pending or active:
                while pending and len(active)<config['workers']:
                    i,b=pending.pop(0);directory=run/'jobs'/rows[i]['audit_id'][:20]/group['builders'][b]['builder']
                    directory.mkdir(parents=True,exist_ok=True)
                    cache=tempfile.TemporaryDirectory(prefix='compile_',dir=run)
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                        NUMEXPR_NUM_THREADS='1',TORCHINDUCTOR_CACHE_DIR=cache.name,TRITON_CACHE_DIR=cache.name+'/triton',
                        TORCHINDUCTOR_FX_GRAPH_CACHE='0',TORCHINDUCTOR_AUTOGRAD_CACHE='0',
                        TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',
                        PYTHONDONTWRITEBYTECODE='1',TMPDIR=cache.name)
                    log=(directory/'train.log').open('w')
                    command=[sys.executable,'-m','structured_kan.scripts.benchmark','--config',str(path),'--worker',str(i),str(b)]
                    proc=subprocess.Popen(command,cwd=PROJECT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                    active.append((proc,log,cache,i,b))
                for job in active[:]:
                    proc,log,cache,i,b=job
                    if proc.poll() is None:continue
                    log.close();cache.cleanup();active.remove(job)
                    if proc.returncode:raise RuntimeError(f'Equation {rows[i]["equation"]}, builder {b+1}: worker exited {proc.returncode}; retained train.log')
                stats=summarize(run,owned_rows)
                update(active=[dict(pid=p.pid,equation=rows[i]['equation'],builder=b+1) for p,_,_,i,b in active],
                       pending_batches=len(pending),completed_fits=stats['completed_fits'],complete_equations=stats['complete_equations'])
                if active:time.sleep(5)
        update(status='complete',active=[],**{k:v for k,v in summarize(run,owned_rows).items() if k!='selected'})
    except BaseException as error:
        update(status='failed',error=repr(error),traceback=traceback.format_exc())
        for proc,log,cache,_,_ in locals().get('active',[]):
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            log.close();cache.cleanup()
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--worker',nargs=2,type=int);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if args.validate_only:
        path,config,rows=inspect_config(args.config)
        count=len(equation_indices(config,rows))
        print(json.dumps(dict(config=str(path),method=config['method'],equations=count,fits=count*18*10,valid=True)));return
    if args.worker is not None:fit_job(args.config,*args.worker)
    else:run_queue(args.config)


if __name__=='__main__':main()
