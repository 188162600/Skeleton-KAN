"""Execute one explicit two-seed PDE shard without changing its frozen config."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

from .benchmark import PROJECT, save, sha
from .pde_benchmark import inspect_config, candidates, candidate_id, fit_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    path, config, problems = inspect_config(manifest['config'])
    assert sha(path) == manifest['configuration_sha256']
    assert config['method'] == 'fourier_mfn'
    first, last = manifest['seed_slice']
    assert first in (0, 2, 4, 6, 8) and last == first + 2
    seeds = config['seeds'][first:last]
    assert seeds == manifest['seeds']
    ei = next(i for i, p in enumerate(problems) if p['id'] == manifest['equation'])
    bi = next(i for i, b in enumerate(candidates(config)) if candidate_id(b) == manifest['builder'])
    run = PROJECT / config['output']
    run.mkdir(parents=True, exist_ok=True)
    job = run / 'jobs' / manifest['equation'] / manifest['builder']
    chunk = job / 'seed_batches' / f'{seeds[0]}_{seeds[-1]}'
    assert not (chunk / 'result.json').exists(), 'Do not overwrite a completed shard'
    state = dict(status='starting', pid=os.getpid(), started=time.time(),
                 method='fourier_mfn', configuration_sha256=sha(path),
                 expected_fits=2, completed_fits=0, effective_workers=1,
                 equations=[manifest['equation']], seeds=seeds,
                 active=[dict(pid=os.getpid(), equation=manifest['equation'],
                              builder=manifest['builder'], seed_batch_size=2)],
                 failed_jobs=[], scientific_protocol_unchanged=True,
                 seed_shard_manifest_sha256=sha(args.manifest))
    save(run / 'state.json', state)
    try:
        with tempfile.TemporaryDirectory(prefix='seed_shard_compile_', dir=run) as cache:
            os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                              OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                              TORCHINDUCTOR_COMPILE_THREADS='4',
                              TORCHINDUCTOR_CACHE_DIR=cache, TRITON_CACHE_DIR=cache+'/triton',
                              TMPDIR=cache, TORCHINDUCTOR_FX_GRAPH_CACHE='0',
                              TORCHINDUCTOR_AUTOGRAD_CACHE='0',
                              TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0',
                              TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE='0',
                              PYTHONDONTWRITEBYTECODE='1',
                              STRUCTURED_KAN_PDE_SEED_BATCH_SIZE='2',
                              STRUCTURED_KAN_PDE_SEED_SLICE=f'{first}:{last}')
            state['status'] = 'training'; save(run / 'state.json', state)
            fit_job(str(path), ei, bi)
        result = json.loads((chunk / 'result.json').read_text())
        assert result['status'] == 'complete'
        assert [s['seed'] for s in result['seeds']] == seeds
        assert result['configuration_sha256'] == sha(path)
        assert result['checkpoint_sha256'] == sha(chunk / 'models.pt')
        state.update(status='complete', completed_fits=2, finished=time.time(),
                     result=str((chunk/'result.json').relative_to(run)))
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished=time.time(),
                     failed_jobs=[manifest['equation']+'/'+manifest['builder']])
        raise
    finally:
        state.update(active=[], updated=time.time())
        save(run / 'state.json', state)


if __name__ == '__main__':
    main()
