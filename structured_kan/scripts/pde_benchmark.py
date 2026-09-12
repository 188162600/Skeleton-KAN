"""Source-300 -> PDE-10: durable remote 18-config x 10-seed PINN sweeps."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from .benchmark import save,sha,PROJECT

def inspect_config(filename):
    path=Path(filename)
    if not path.is_file():path=PROJECT/'data/train_config'/filename
    c=json.loads(path.read_text());assert c['schema']=='structured-kan.pde-training.v1'
    assert c['seeds']==list(range(421,431)) and c['points_per_split']==10000
    assert (c['outer_steps'],c['max_inner_iterations'],c['history_size'])==(50,20,100)
    assert c['interior_solution_in_training'] is False
    if c['method']=='smpf_structure':
        from ..structure_builder.smpf import architecture_sweep
        assert c['architectures']==architecture_sweep(input_mode=c.get('input_mode','full_affine')) and c['report_floor']==0
        template=PROJECT/c['template']['path']
        assert sha(template)==c['template']['sha256']
    if c['method']=='v2':
        assert sha(PROJECT/c['catalogue'])==c['catalogue_sha256']
        assert sha(PROJECT/c['construction'])==c['construction_sha256']
        assert c['materialization'] in ('shared-affine-private-phi-v1','private-affine-private-phi-v1','private-affine-shared-phi-v1')
        assert c['phi_sharing_policy']=='source-role-compatible-G-and-prior-v1'
        if c['materialization']=='private-affine-private-phi-v1':
            assert c['paired_initialization']=='shared-affine-private-phi-v1'
    assert sha(PROJECT/c['source']['path'])==c['source']['sha256']
    assert len(json.loads((PROJECT/c['source']['path']).read_text()))==300
    assert sha(PROJECT/c['problems']['path'])==c['problems']['sha256']
    ps=json.loads((PROJECT/c['problems']['path']).read_text())['problems'];assert len(ps)==10
    return path.resolve(),c,ps

def prepare(config):
    if config['method'] in ('mlp','multkan','fourier_mfn','smpf_structure'):return
    src=PROJECT/config['source']['path'];base=PROJECT/config['construction'];dest=PROJECT/config['catalogue']
    group=json.loads(base.read_text())
    assert group['training_equations']==300 and group['training_sha256']==sha(src)
    assert group['heldout_domain']=='pde' and group['builder_count']==18
    assert group['construction']['existing_catalogue_used'] is False
    if not dest.exists():
        if config['method']=='v2':
            from ..structure_builder.shared_affine_complexity import allocate
        else:
            from ..structure_builder.complexity import allocate
        save(dest,allocate(base,src))
    allocated=json.loads(dest.read_text())
    assert allocated['complexity_allocation']['source_rows']==300
    assert allocated['complexity_allocation']['base_catalogue_sha256']==sha(base)
    assert allocated['complexity_allocation']['heldout_rows_read']==0
    if config['method']=='v2':
        assert allocated['builder_encoding']=='shared-reserve-constructor-records-v1'
        assert allocated['method']=='v2' and allocated['source_only']

def candidates(config):
    if config['method'] in ('mlp','multkan','fourier_mfn','smpf_structure'):return config['architectures']
    return json.loads((PROJECT/config['catalogue']).read_text())['builders']

def candidate_id(candidate):return candidate.get('builder',candidate.get('id'))

def make_model(config,problem,candidate,seed,data,device='cuda'):
    from ..model.MLP import MLP,parameter_hash
    from ..model.StructuredKANBuilder import StructuredKANBuilder
    from ..model.topology_spec import finite_topology_spec
    from ..model.initialization import perturb_trainable_
    dim=len(problem['coordinates'])
    if config['method']=='mlp':
        spec=candidate
        model=MLP(dim,candidate['hidden_layers'],candidate['hidden_width'],candidate['activation'],seed=seed,device=device)
        initial=parameter_hash(model)
    elif config['method'] in ('multkan','fourier_mfn','smpf_structure'):
        from ..model.neural_baselines import build_model
        spec=candidate
        inputs=(data['interior']-data['x_mean'])/data['x_std'] if config['method']=='smpf_structure' else None
        model=build_model(config['method'],dim,candidate,seed=seed,device=device,train_inputs=inputs)
        if config['method']=='smpf_structure':spec=model.structure_specification
        initial=parameter_hash(model)
    else:
        paired_spec=None;materialization=None
        if config['method']=='v2':
            from ..model.shared_affine_spec import shared_affine_private_phi_spec,private_affine_shared_phi_spec
            mode=config['materialization']
            if mode=='private-affine-shared-phi-v1':
                spec,materialization=private_affine_shared_phi_spec(candidate,dim,off_diagonal=config['route_off_diagonal'])
            else:
                shared=mode=='shared-affine-private-phi-v1'
                spec,materialization=shared_affine_private_phi_spec(candidate,dim,off_diagonal=config['route_off_diagonal'],share_affine=shared)
                if not shared:
                    paired_spec,_=shared_affine_private_phi_spec(candidate,dim,off_diagonal=config['route_off_diagonal'])
        else:
            budgets=candidate['phi_basis_by_label']
            spec=finite_topology_spec(candidate['synthetic_operator_tree'],dim,per_map_G=budgets,
                output_G=budgets['phi_'+str(len(budgets)-1)],off_diagonal=config['route_off_diagonal'])
        def explicit_products(node):
            if node.get('operation')=='prod':node['product_reduction']='sequential'
            for child in node.get('children',[]):explicit_products(child)
        explicit_products(spec)
        if paired_spec is not None:explicit_products(paired_spec)
        inputs=(data['interior']-data['x_mean'])/data['x_std']
        model=StructuredKANBuilder(dim,device=device).build(spec,
            train_inputs=inputs)
        if paired_spec is None:
            initial=perturb_trainable_(model,seed,config['initialization_noise'])
        else:
            from ..model.shared_affine_spec import initialize_private_from_shared_
            initial,_=initialize_private_from_shared_(model,paired_spec,inputs,seed,config['initialization_noise'])
        if materialization is not None:
            assert model.parameter_count()==materialization['expected_parameters']
            model.materialization_metadata=materialization
    return model,spec,initial

def fit_job(filename,equation_index,builder_index):
    batch_size=int(os.environ.get('STRUCTURED_KAN_PDE_SEED_BATCH_SIZE','10'))
    seed_slice=os.environ.get('STRUCTURED_KAN_PDE_SEED_SLICE')
    assert batch_size in (1,2,5,10)
    if batch_size<10 and not seed_slice:
        from .pde_seed_chunks import run_split_job
        return run_split_job(filename,equation_index,builder_index,batch_size)
    import torch
    from ..dataset.pde import sample
    from ..dataset.regression import regression_metrics
    from ..optimizer.pde_fit import fit_pde_seed_batch
    path,c,ps=inspect_config(filename);p=ps[equation_index];candidate=candidates(c)[builder_index]
    run=PROJECT/c['output'];job=run/'jobs'/p['id']/candidate_id(candidate)
    if seed_slice:
        first,last=map(int,seed_slice.split(':'))
        assert batch_size in (1,2,5) and first % batch_size == 0 and last == first + batch_size and 0 <= first < last <= 10
        c=dict(c,seeds=c['seeds'][first:last])
        job=job/'seed_batches'/f'{c["seeds"][0]}_{c["seeds"][-1]}'
    job.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    assert not (job/'result.json').exists(),'Completed results must not be overwritten'
    started=time.time();data=[];evaluation=[];receipts=[];models=[];initial=[]
    for seed in c['seeds']:
        d,e,r=sample(p,seed,points=c['points_per_split'],boundary_points=c['boundary_points'],initial_points=c['initial_points'],device='cuda')
        model,spec,h=make_model(c,p,candidate,seed,d)
        data.append(d);evaluation.append(e);receipts.append(r);models.append(model);initial.append(h)
    assert len(set(initial))==len(c['seeds'])
    parameters=sum(q.numel() for q in models[0].parameters() if q.requires_grad)
    frozen=[{n:t.detach().clone() for n,t in model.named_buffers()} for model in models]
    def progress(outer,loss,iterations,evaluations):
        save(job/'progress.json',dict(status='training',pid=os.getpid(),equation=p['id'],builder=candidate_id(candidate),
            outer=outer,outer_steps=50,seconds=time.time()-started,training_pinn_loss=loss.cpu().tolist(),
            inner_iterations=iterations,function_evaluations=evaluations))
    save(job/'progress.json',dict(status='compiling',pid=os.getpid(),equation=p['id'],builder=candidate_id(candidate),parameters=parameters,started=started))
    fit=fit_pde_seed_batch(models,data,p,outer_steps=c['outer_steps'],compile_mode=c['compile_mode'],progress=progress)
    assert all(torch.equal(t,dict(m.named_buffers())[n]) for m,b in zip(models,frozen) for n,t in b.items())
    records=[];states=[]
    for index,(m,d,e) in enumerate(zip(models,data,evaluation)):
        metrics={}
        with torch.no_grad():
            for split in ('validation','test'):
                prediction=m((e[split]['x']-d['x_mean'])/d['x_std'])*d['y_std']+d['y_mean']
                metrics[split]=regression_metrics(prediction,e[split]['y'])
                assert all(math.isfinite(v) and v>=0 for v in metrics[split].values())
        records.append(dict(seed=c['seeds'][index],equation=p['id'],method=c['method'],builder=candidate_id(candidate),
            parameters=parameters,metrics=metrics,training_pinn_loss=float(fit.loss[index]),
            initialization_sha256=initial[index],inner_iterations=fit.inner_iterations[index],function_evaluations=fit.function_evaluations[index]))
        states.append({n:t.detach().cpu() for n,t in m.state_dict().items()})
    checkpoint=job/'models.pt';temp=job/'models.tmp'
    normals=[{k:d[k].detach().cpu() for k in ('x_mean','x_std','y_mean','y_std')} for d in data]
    torch.save(dict(schema='structured-kan.pde-checkpoint.v1',structure=spec,states=states,problem=p,seeds=c['seeds'],
        normalization=normals,results=records,data_receipts=receipts,configuration_sha256=sha(path),
        materialization=getattr(models[0],'materialization_metadata',None)),temp);temp.replace(checkpoint)
    # Replay the stored checkpoint before marking this batch complete.
    replay=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert len(replay['states'])==len(c['seeds']) and all(torch.equal(replay['states'][i][n],v) for i,s in enumerate(states) for n,v in s.items())
    result=dict(status='complete',equation=p['id'],builder=candidate_id(candidate),method=c['method'],parameters=parameters,
        seeds=records,seconds=time.time()-started,finished=time.time(),
        mean_log_validation_nmse=sum(math.log(max(r['metrics']['validation']['nmse'],1e-300)) for r in records)/len(records),
        configuration_sha256=sha(path),checkpoint_sha256=sha(checkpoint),data_receipts=receipts,
        materialization=getattr(models[0],'materialization_metadata',None))
    save(job/'result.json',result);save(job/'progress.json',dict(status='complete',seconds=result['seconds'],finished=result['finished']))
    print(json.dumps({k:result[k] for k in ('equation','builder','parameters','seconds')}),flush=True)

def summarize(run,ps):
    selected=[];count=0
    for p in ps:
        rows=[json.loads(f.read_text()) for f in (run/'jobs'/p['id']).glob('*/result.json')];count+=len(rows)
        if len(rows)!=18:continue
        best=min(rows,key=lambda r:(r['mean_log_validation_nmse'],r['builder']))
        record=dict(equation=p['id'],builder=best['builder'],parameters=best['parameters'])
        for metric in ('mse','nmse'):
            record['test_g'+metric]=math.exp(sum(math.log(max(s['metrics']['test'][metric],1e-10)) for s in best['seeds'])/10)
        selected.append(record)
    out=dict(completed_batches=count,completed_fits=count*10,expected_fits=len(ps)*180,complete_equations=len(selected),selected=selected,
        selection='one config per PDE, mean log validation NMSE over ten seeds',report_floor=1e-10)
    if selected:
        out['mean_selected_parameters']=sum(q['parameters'] for q in selected)/len(selected)
        for metric in ('test_gmse','test_gnmse'):out[metric]=math.exp(sum(math.log(q[metric]) for q in selected)/len(selected))
    save(run/'summary.json',out);return out

def run_queue(filename):
    from .pde_failures import FailureLedger
    path,c,ps=inspect_config(filename);run=PROJECT/c['output'];run.mkdir(parents=True,exist_ok=True)
    report_summarize = summarize
    if c['method']=='smpf_structure':
        from .smpf_report import summarize_pde as report_summarize
    lock=run/'queue.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    state=dict(status='preparing',pid=os.getpid(),started=time.time(),method=c['method'],configuration_sha256=sha(path),source_equations=300)
    # Execution-only override: preserve the scientific configuration hash and
    # every completed checkpoint when reducing GPU concurrency after an OOM.
    workers=int(os.environ.get('STRUCTURED_KAN_PDE_WORKERS',c['workers']))
    assert 1<=workers<=c['workers']
    shard_count=int(os.environ.get('STRUCTURED_KAN_PDE_SHARDS','1'))
    shard_index=int(os.environ.get('STRUCTURED_KAN_PDE_SHARD','0'))
    assert 1<=shard_count<=len(ps) and 0<=shard_index<shard_count
    shard_problems=[p for i,p in enumerate(ps) if i%shard_count==shard_index]
    state.update(shard_index=shard_index,shard_count=shard_count,
        equations=[p['id'] for p in shard_problems],expected_fits=180*len(shard_problems))
    state['effective_workers']=workers
    state['execution_environment']={k:os.environ[k] for k in (
        'TORCHINDUCTOR_PERSISTENT_REDUCTIONS','STRUCTURED_KAN_PDE_SKIP_FAILED',
        'STRUCTURED_KAN_PDE_RETRY_FAILED','STRUCTURED_KAN_PDE_SEED_BATCH_SIZE',
        'STRUCTURED_KAN_PDE_CUDAGRAPHS','STRUCTURED_KAN_PDE_REAL_FX',
        'STRUCTURED_KAN_PDE_FORWARD_AD') if k in os.environ}
    skip_failed=os.environ.get('STRUCTURED_KAN_PDE_SKIP_FAILED')=='1'
    retry_failed=os.environ.get('STRUCTURED_KAN_PDE_RETRY_FAILED')=='1'
    failures=FailureLedger(run,sha(path))
    state['failure_policy']='defer_and_continue' if skip_failed else 'stop'
    active=[]
    def update(**kw):
        unresolved=failures.unresolved()
        state.update(kw,updated=time.time(),failed_jobs=unresolved,
                     failed_batches=len(unresolved),deferred_fits=len(unresolved)*len(c['seeds']))
        save(run/'state.json',state)
    update()
    try:
        prepare(c);items=candidates(c);assert len(items)==18
        pending=[]
        for i,p in enumerate(ps):
            if i%shard_count!=shard_index:continue
            for j,b in enumerate(items):
                folder=run/'jobs'/p['id']/candidate_id(b);result=folder/'result.json'
                if result.exists():
                    saved=json.loads(result.read_text());assert saved['configuration_sha256']==sha(path)
                    assert saved['checkpoint_sha256']==sha(folder/'models.pt')
                elif not (skip_failed and not retry_failed and p['id']+'/'+candidate_id(b) in failures.unresolved()):
                    pending.append((i,j))
        while pending or active:
            while pending and len(active)<workers:
                i,j=pending.pop(0);folder=run/'jobs'/ps[i]['id']/candidate_id(items[j]);folder.mkdir(parents=True,exist_ok=True)
                cache=tempfile.TemporaryDirectory(prefix='compile_',dir=run)
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                    TORCHINDUCTOR_COMPILE_THREADS='4',TORCHINDUCTOR_CACHE_DIR=cache.name,TRITON_CACHE_DIR=cache.name+'/triton',
                    TORCHINDUCTOR_FX_GRAPH_CACHE='0',TORCHINDUCTOR_AUTOGRAD_CACHE='0',TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',
                    TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',TMPDIR=cache.name,PYTHONDONTWRITEBYTECODE='1')
                # Never discard the failure trace or an interrupted progress record.
                if (folder/'train.log').exists():
                    archive=folder/('attempt_'+str(time.time_ns()));archive.mkdir()
                    for name in ('train.log','progress.json'):
                        if (folder/name).exists():(folder/name).rename(archive/name)
                log=(folder/'train.log').open('w')
                command=[sys.executable,'-m','structured_kan.scripts.pde_benchmark','--config',str(path),'--worker',str(i),str(j)]
                proc=subprocess.Popen(command,cwd=PROJECT.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
                active.append((proc,log,cache,i,j))
            for entry in active[:]:
                proc,log,cache,i,j=entry
                if proc.poll() is None:continue
                log.close();cache.cleanup();active.remove(entry)
                if proc.returncode:
                    error=f'{ps[i]["id"]}/{candidate_id(items[j])}: worker exited {proc.returncode}; see train.log'
                    if not skip_failed:raise RuntimeError(error)
                    failures.record(ps[i]['id'],candidate_id(items[j]),proc.returncode,error)
            stats=report_summarize(run,shard_problems)
            update(status='training',pending_batches=len(pending),active=[dict(pid=q.pid,equation=ps[i]['id'],builder=candidate_id(items[j])) for q,_,_,i,j in active],
                completed_fits=stats['completed_fits'],complete_equations=stats['complete_equations'])
            if active:time.sleep(5)
        update(status='complete_with_failures' if failures.unresolved() else 'complete',
               active=[],**{k:v for k,v in report_summarize(run,shard_problems).items() if k!='selected'})
    except BaseException as error:
        update(status='failed',error=repr(error),traceback=traceback.format_exc())
        for proc,log,cache,_,_ in active:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
            log.close();cache.cleanup()
        raise
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--worker',nargs=2,type=int);p.add_argument('--prepare',action='store_true')
    a=p.parse_args()
    if a.prepare:
        _,c,_=inspect_config(a.config);prepare(c);print(json.dumps(dict(status='allocated',method=c['method'],source_rows=300)));return
    if a.worker is None:run_queue(a.config)
    else:fit_job(a.config,*a.worker)
if __name__=='__main__':main()
