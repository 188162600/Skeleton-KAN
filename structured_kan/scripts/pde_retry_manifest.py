"""Run explicit failed PDE jobs, preserving ten seeds and the scientific budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .benchmark import PROJECT, save, sha
from .pde_benchmark import inspect_config, candidates, candidate_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cfgpath, config, problems = inspect_config(manifest['config'])
    assert sha(cfgpath) == manifest['configuration_sha256']
    assert config['method'] == 'fourier_mfn'
    run = PROJECT / config['output']
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'queue.lock'
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    items = candidates(config)
    indices = {p['id']: i for i, p in enumerate(problems)}
    builder_indices = {candidate_id(b): j for j, b in enumerate(items)}
    jobs = manifest['jobs']
    state = dict(status='training', pid=os.getpid(), started=time.time(), method='fourier_mfn',
                 configuration_sha256=sha(cfgpath), expected_fits=10 * len(jobs),
                 equations=sorted({j['equation'] for j in jobs}), retry_jobs=jobs,
                 effective_workers=1, active=[], failed_jobs=[], completed_fits=0,
                 attempts=[], scientific_protocol_unchanged=True)
    child = None

    def update():
        rows = []
        for job in jobs:
            p = run / 'jobs' / job['equation'] / job['builder'] / 'result.json'
            if p.exists():
                row = json.loads(p.read_text())
                assert row['status'] == 'complete' and len(row['seeds']) == 10
                assert row['configuration_sha256'] == sha(cfgpath)
                assert row['checkpoint_sha256'] == sha(p.parent / 'models.pt')
                rows.append(row)
        state.update(updated=time.time(), completed_fits=10 * len(rows),
                     completed_retry_jobs=len(rows), failed_batches=len(state['failed_jobs']),
                     deferred_fits=10 * len(state['failed_jobs']))
        save(run / 'state.json', state)

    try:
        update()
        for n, job in enumerate(jobs):
            directory = run / 'jobs' / job['equation'] / job['builder']
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'result.json').exists():
                continue
            sizes = [b for b in (5, 2, 1) if b <= job['seed_batch_size']]
            for size in sizes:
                with tempfile.TemporaryDirectory(prefix='retry_compile_', dir=run) as cache:
                    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                               TORCHINDUCTOR_COMPILE_THREADS='4', TORCHINDUCTOR_CACHE_DIR=cache,
                               TRITON_CACHE_DIR=cache + '/triton', TMPDIR=cache,
                               TORCHINDUCTOR_FX_GRAPH_CACHE='0', TORCHINDUCTOR_AUTOGRAD_CACHE='0',
                               TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',
                               TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0', PYTHONDONTWRITEBYTECODE='1',
                               STRUCTURED_KAN_PDE_SEED_BATCH_SIZE=str(size))
                    env.pop('STRUCTURED_KAN_PDE_SEED_SLICE', None)
                    logfile = directory / f'retry_batch{size}_{time.time_ns()}.log'
                    command = [sys.executable, '-m', 'structured_kan.scripts.pde_benchmark',
                               '--config', str(cfgpath), '--worker', str(indices[job['equation']]),
                               str(builder_indices[job['builder']])]
                    with logfile.open('w') as log:
                        child = subprocess.Popen(command, cwd=PROJECT.parent, env=env,
                                                 stdin=subprocess.DEVNULL, stdout=log,
                                                 stderr=subprocess.STDOUT)
                        state['active'] = [dict(pid=child.pid, equation=job['equation'],
                                                builder=job['builder'], seed_batch_size=size)]
                        state['pending_batches'] = len(jobs) - n - 1
                        while child.poll() is None:
                            update(); time.sleep(5)
                        rc = child.returncode
                    state['attempts'].append(dict(job=job['equation'] + '/' + job['builder'],
                                                  seed_batch_size=size, returncode=rc,
                                                  log=str(logfile), finished=time.time()))
                    child = None
                state['active'] = []
                if rc == 0:
                    update(); break
                # Only retry at lower independent-seed concurrency for an OOM.
                texts = [logfile.read_text(errors='replace')[-12000:]]
                texts += [p.read_text(errors='replace')[-12000:]
                          for p in directory.glob('seed_batches/*/attempt_*.log')]
                oom = any('out of memory' in s.lower() or 'OutOfMemoryError' in s for s in texts)
                if not oom or size == 1:
                    state['failed_jobs'].append(job['equation'] + '/' + job['builder'])
                    break
            update()
        state['status'] = 'complete_with_failures' if state['failed_jobs'] else 'complete'
        state['active'] = []; update()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        state.update(status='failed', error=repr(error)); update()
        raise
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    main()
