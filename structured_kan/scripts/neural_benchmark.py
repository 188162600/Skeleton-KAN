"""Remote Domain300 neural baselines using one common sampler and optimizer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

from .benchmark import PROJECT, DOMAINS, save, sha, summarize, equation_indices

SCHEMAS = {'mlp': 'structured-kan.mlp-domain80.v1', 'fourier_mfn': 'structured-kan.fourier-mfn-domain80.v1',
           'multkan': 'structured-kan.multkan-domain80.v1',
           'smpf_structure': 'structured-kan.smpf-structure-domain80.v1'}


def inspect_config(filename):
    path = Path(filename)
    if not path.is_file():
        path = PROJECT / 'data/train_config' / filename
    config = json.loads(path.read_text())
    assert config['schema'] == SCHEMAS[config['method']]
    assert config['seeds'] == list(range(421, 431))
    assert config['points_per_split'] == 10000
    assert (config['outer_steps'], config['max_inner_iterations'], config['history_size']) == (50, 20, 100)
    assert config['line_search'] == 'strong-Wolfe'
    assert config['normalization'] == 'train-only-standardization'
    assert config['data_domains'] == list(DOMAINS)
    assert config['workers'] == 2 and config['save_models']
    assert config['report_floor'] == (0 if config['method']=='smpf_structure' else 1e-10)
    from ..model.neural_baselines import architecture_sweep
    if config['method']=='smpf_structure':
        from ..structure_builder.smpf import architecture_sweep as smpf_sweep
        assert config['architectures'] == smpf_sweep(input_mode=config.get('input_mode','full_affine'))
    else:
        assert config['architectures'] == architecture_sweep(config['method'])
    subset, parent = (PROJECT / config[k]['path'] for k in ('subset', 'parent'))
    assert sha(subset) == config['subset']['sha256']
    assert sha(parent) == config['parent']['sha256']
    rows, parents = json.loads(subset.read_text()), json.loads(parent.read_text())
    assert len(rows) == 80 and len(parents) == 300
    source = {r['case_id']: r for r in parents}
    assert len(source) == 300 and len({r['case_id'] for r in rows}) == 80
    for row in rows:
        for key in ('source_corpus', 'variables', 'analysis_expression', 'operator_variable_spec', 'quality_audit'):
            assert row[key] == source[row['case_id']][key]
    assert all(sum(r['source_corpus'] == d for r in rows) == 20 for d in DOMAINS)
    equation_indices(config, rows)
    if config['method']=='smpf_structure':
        template=PROJECT/config['template']['path']
        assert sha(template)==config['template']['sha256']
        assert json.loads(template.read_text())['architectures']==config['architectures']
    return path.resolve(), config, rows


def fit_job(config_path, equation_index, architecture_index):
    import torch
    from ..dataset.fresh_loader import load_seed_data
    from ..dataset.regression import regression_metrics
    from ..model.neural_baselines import build_model, parameter_hash, expected_parameters
    from ..optimizer.fit import fit_seed_batch
    path, config, rows = inspect_config(config_path)
    assert equation_index in equation_indices(config, rows)
    run = PROJECT / config['output']
    row, arch = rows[equation_index], config['architectures'][architecture_index]
    job = run / 'jobs' / row['audit_id'][:20] / arch['id']
    job.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available(), 'CUDA required, no CPU fallback'
    raw, receipt = load_seed_data(row, run / 'sampling')
    standardized = [data.standardize() for data in raw]
    datasets, normalizations = zip(*standardized)
    models = [build_model(config['method'], row['variables'], arch, seed=seed, device='cuda',
              train_inputs=datasets[i].train.x if config['method']=='smpf_structure' else None)
              for i,seed in enumerate(config['seeds'])]
    frozen=[{n:t.detach().clone() for n,t in m.named_buffers()} for m in models]
    initial = [parameter_hash(model) for model in models]
    assert len(set(initial)) == 10
    parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    assert parameters == expected_parameters(config['method'], row['variables'], arch)
    def progress(outer, loss, iterations, evaluations):
        save(job / 'progress.json', dict(status='training', pid=os.getpid(), equation=row['equation'],
             builder=arch['id'], outer=outer, outer_steps=50, seconds=time.time()-started,
             train_standardized_mse=loss.cpu().tolist(), inner_iterations=iterations,
             function_evaluations=evaluations))
    save(job / 'progress.json', dict(status='compiling', pid=os.getpid(), equation=row['equation'],
         builder=arch['id'], parameters=parameters, started=started))
    fit = fit_seed_batch(models, datasets, outer_steps=50, compile_mode=config['compile_mode'], progress=progress)
    assert all(torch.equal(t,dict(m.named_buffers())[n]) for m,b in zip(models,frozen) for n,t in b.items())
    records, states = [], []
    for index, (model, data, norm) in enumerate(zip(models, datasets, normalizations)):
        metrics = {}
        with torch.no_grad():
            for split in ('train', 'validation', 'test'):
                prediction = model(getattr(data, split).x.cuda()) * norm['y_std'].cuda() + norm['y_mean'].cuda()
                metrics[split] = regression_metrics(prediction, getattr(raw[index], split).y.cuda())
                assert all(math.isfinite(v) and v >= 0 for v in metrics[split].values()), metrics
        records.append(dict(seed=config['seeds'][index], case_id=row['case_id'], equation=row['equation'],
            domain=row['source_corpus'], method=config['method'], builder=arch['id'], architecture=arch,
            parameters=parameters, metrics=metrics, initialization_sha256=initial[index],
            inner_iterations=fit.inner_iterations[index], function_evaluations=fit.function_evaluations[index]))
        states.append({n: t.detach().cpu() for n, t in model.state_dict().items()})
    checkpoint, temporary = job / 'models.pt', job / 'models.tmp'
    torch.save(dict(schema='structured-kan.neural-seed-batch-checkpoint.v1', method=config['method'], input_dim=row['variables'],
        architecture=arch, states=states, seeds=config['seeds'], normalization=list(normalizations),
        results=records, configuration_sha256=sha(path), data_receipt=receipt,
        structure=getattr(models[0],'structure_specification',None)), temporary)
    temporary.replace(checkpoint)
    result = dict(status='complete', equation=row['equation'], case_id=row['case_id'], domain=row['source_corpus'],
        builder=arch['id'], architecture=arch, method=config['method'], parameters=parameters, seeds=records,
        seconds=time.time()-started, finished=time.time(), configuration_sha256=sha(path),
        checkpoint_sha256=sha(checkpoint),
        mean_log_validation_nmse=sum(math.log(max(r['metrics']['validation']['nmse'], 1e-300)) for r in records)/10)
    save(job / 'result.json', result)
    save(job / 'progress.json', dict(status='complete', seconds=result['seconds'], finished=result['finished']))
    print(json.dumps(dict(equation=row['equation'], configuration=arch['id'], seconds=result['seconds'], parameters=parameters)), flush=True)


def run_queue(config_path):
    from ..dataset.fresh_sampling import generate_row
    from ..dataset.fresh_loader import load_seed_data
    path, config, rows = inspect_config(config_path)
    owned=equation_indices(config, rows);owned_rows=[rows[i] for i in owned]
    run = PROJECT / config['output']
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'queue.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    state = dict(status='preparing', pid=os.getpid(), method=config['method'], started=time.time(),
                 total_fits=len(owned)*180, total_batches=len(owned)*18, configuration_sha256=sha(path))
    report_summarize = summarize
    if config['method']=='smpf_structure':
        from .smpf_report import summarize_transfer as report_summarize
    def update(**values):
        state.update(values, updated=time.time())
        save(run / 'state.json', state)
    update()
    active = []
    try:
        for domain in DOMAINS:
            indices = [i for i in owned if rows[i]['source_corpus'] == domain]
            update(status='sampling', domain=domain, sampling_completed=0, sampling_expected=len(indices))
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(generate_row, (rows[i], run / 'sampling', 10000)) for i in indices]
                for completed, future in enumerate(as_completed(futures), 1):
                    receipt = future.result()
                    assert receipt['status'] == 'complete', receipt
                    update(sampling_completed=completed, last_sample=receipt['case_id'])
            for i in indices:
                load_seed_data(rows[i], run / 'sampling')
            pending = []
            for i in indices:
                for b, arch in enumerate(config['architectures']):
                    result = run / 'jobs' / rows[i]['audit_id'][:20] / arch['id'] / 'result.json'
                    if result.exists():
                        saved = json.loads(result.read_text())
                        assert saved['configuration_sha256'] == sha(path)
                        assert saved['checkpoint_sha256'] == sha(result.with_name('models.pt'))
                    else:
                        pending.append((i, b))
            update(status='training', pending_batches=len(pending), active=[])
            while pending or active:
                while pending and len(active) < config['workers']:
                    i, b = pending.pop(0)
                    directory = run / 'jobs' / rows[i]['audit_id'][:20] / config['architectures'][b]['id']
                    directory.mkdir(parents=True, exist_ok=True)
                    cache = tempfile.TemporaryDirectory(prefix='compile_', dir=run)
                    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                        NUMEXPR_NUM_THREADS='1', TORCHINDUCTOR_CACHE_DIR=cache.name, TRITON_CACHE_DIR=cache.name+'/triton',
                        TORCHINDUCTOR_FX_GRAPH_CACHE='0', TORCHINDUCTOR_AUTOGRAD_CACHE='0',
                        TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0', TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',
                        PYTHONDONTWRITEBYTECODE='1', TMPDIR=cache.name, CUDA_CACHE_PATH=cache.name+'/cuda',
                        TORCHINDUCTOR_COMPILE_THREADS='2', MALLOC_ARENA_MAX='2')
                    log = (directory / 'train.log').open('w')
                    command = [sys.executable, '-m', 'structured_kan.scripts.neural_benchmark', '--config', str(path), '--worker', str(i), str(b)]
                    proc = subprocess.Popen(command, cwd=PROJECT.parent, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active.append((proc, log, cache, i, b))
                for item in active[:]:
                    proc, log, cache, i, b = item
                    if proc.poll() is None:
                        continue
                    log.close()
                    cache.cleanup()
                    active.remove(item)
                    if proc.returncode:
                        raise RuntimeError(f'Equation {rows[i]["equation"]}, config {b+1}: exit {proc.returncode}; inspect train.log')
                stats = report_summarize(run, owned_rows)
                update(active=[dict(pid=p.pid, equation=rows[i]['equation'], builder=b+1) for p, _, _, i, b in active],
                       pending_batches=len(pending), completed_fits=stats['completed_fits'], complete_equations=stats['complete_equations'])
                if active:
                    time.sleep(5)
        update(status='complete', active=[], **{k: v for k, v in report_summarize(run, owned_rows).items() if k != 'selected'})
    except BaseException as error:
        update(status='failed', error=repr(error), traceback=traceback.format_exc())
        for proc, log, cache, _, _ in active:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            log.close()
            cache.cleanup()
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--worker', nargs=2, type=int)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        path, config, rows = inspect_config(args.config)
        print(json.dumps(dict(config=str(path), method=config['method'], equations=len(rows), fits=14400, valid=True)))
    elif args.worker is not None:
        fit_job(args.config, *args.worker)
    else:
        run_queue(args.config)


if __name__ == '__main__':
    main()
