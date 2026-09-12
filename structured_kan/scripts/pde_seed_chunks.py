"""Run independent PDE seed groups in fresh processes and join all ten safely."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .benchmark import save, sha, PROJECT


def run_split_job(filename, equation_index, builder_index, batch_size):
    from .pde_benchmark import inspect_config, candidates, candidate_id
    path, config, problems = inspect_config(filename)
    assert config['method'] in ('mlp', 'fourier_mfn') and batch_size in (1, 2, 5)
    assert len(config['seeds']) == 10 and 10 % batch_size == 0
    problem = problems[equation_index]
    builder = candidate_id(candidates(config)[builder_index])
    job = PROJECT / config['output'] / 'jobs' / problem['id'] / builder
    job.mkdir(parents=True, exist_ok=True)
    assert not (job / 'result.json').exists()
    started = time.time()
    folders = []
    active = None

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for first in range(0, len(config['seeds']), batch_size):
            last = first + batch_size
            seeds = config['seeds'][first:last]
            folder = job / 'seed_batches' / f'{seeds[0]}_{seeds[-1]}'
            folder.mkdir(parents=True, exist_ok=True)
            folders.append(folder)
            if (folder / 'result.json').exists():
                saved = json.loads((folder / 'result.json').read_text())
                assert saved['configuration_sha256'] == sha(path)
                assert saved['checkpoint_sha256'] == sha(folder / 'models.pt')
                assert [r['seed'] for r in saved['seeds']] == seeds
                continue
            env = dict(os.environ, STRUCTURED_KAN_PDE_SEED_SLICE=f'{first}:{last}',
                       STRUCTURED_KAN_PDE_SEED_BATCH_SIZE=str(batch_size))
            command = [sys.executable, '-m', 'structured_kan.scripts.pde_benchmark',
                       '--config', str(path), '--worker', str(equation_index), str(builder_index)]
            with (folder / f'attempt_{time.time_ns()}.log').open('w') as log:
                active = subprocess.Popen(command, cwd=PROJECT.parent, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                while active.poll() is None:
                    progress = folder / 'progress.json'
                    item = json.loads(progress.read_text()) if progress.exists() else dict(status='starting')
                    item.update(seed_batch_size=batch_size, batch_seeds=seeds, batch_number=first//batch_size+1,
                                seed_batches=10//batch_size, coordinator_pid=os.getpid(), chunk_pid=active.pid,
                                equation=problem['id'], builder=builder)
                    save(job / 'progress.json', item)
                    time.sleep(3)
                if active.returncode:
                    raise RuntimeError(f'{problem["id"]}/{builder} seeds {seeds}: '
                                       f'chunk exited {active.returncode}; see {folder}')
                active = None
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                active.kill()
                active.wait(timeout=10)
        signal.signal(signal.SIGTERM, previous)

    # Only CPU state loading here; each GPU child has exited and released memory.
    import torch
    results = [json.loads((f / 'result.json').read_text()) for f in folders]
    for f, result in zip(folders, results):
        assert result['configuration_sha256'] == sha(path)
        assert result['checkpoint_sha256'] == sha(f / 'models.pt')
        assert result['status'] == 'complete'
    chunks = [torch.load(f / 'models.pt', map_location='cpu', weights_only=True) for f in folders]
    records = [v for r in results for v in r['seeds']]
    assert [r['seed'] for r in records] == config['seeds']
    assert len({r['initialization_sha256'] for r in records}) == 10
    assert all(r['parameters'] == results[0]['parameters'] for r in results)
    assert all(c['structure'] == chunks[0]['structure'] and c['problem'] == problem for c in chunks)
    checkpoint = dict(chunks[0])
    for field in ('states', 'normalization', 'results', 'data_receipts', 'seeds'):
        checkpoint[field] = [v for c in chunks for v in c[field]]
    assert checkpoint['seeds'] == config['seeds'] and checkpoint['results'] == records
    checkpoint['execution'] = dict(seed_batch_size=batch_size, total_seeds=10, fresh_process_per_batch=True)
    temporary = job / 'models.tmp'
    torch.save(checkpoint, temporary)
    temporary.replace(job / 'models.pt')
    replay = torch.load(job / 'models.pt', map_location='cpu', weights_only=True)
    assert all(torch.equal(replay['states'][i][n], v)
               for i, state in enumerate(checkpoint['states']) for n, v in state.items())
    result = dict(results[0], seeds=records, data_receipts=checkpoint['data_receipts'],
                  seconds=time.time()-started, finished=time.time(),
                  mean_log_validation_nmse=sum(math.log(max(r['metrics']['validation']['nmse'],1e-300)) for r in records)/10,
                  checkpoint_sha256=sha(job / 'models.pt'), execution=checkpoint['execution'],
                  seed_batch_results=[dict(path=str(f.relative_to(job)), sha256=sha(f/'result.json')) for f in folders])
    save(job / 'result.json', result)
    save(job / 'progress.json', dict(status='complete', seconds=result['seconds'],
                                   seed_batch_size=batch_size, total_seeds=10, finished=result['finished']))
