"""Remote-only fresh sampling: ten independent train/validation/test triples.

Box tasks retain their audited probability laws. Continuous trajectory tasks
use independently sampled observation times on the same source horizon, not
new initial conditions and not interpolation of the previous 30k-point pool.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'domain300-fresh-thirty-sets-v2'
SEEDS = tuple(range(421, 431))
SPLITS = ('train', 'validation', 'test')
COMMIT = '6a09daf46af1bb89e4857436b623a7b8720863ad'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    import numpy as np
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda a: a.tolist() if hasattr(a, 'tolist') else str(a)))
    tmp.replace(path)


def sampling_seed(case_id, seed, split):
    assert split in SPLITS
    text = f'{VERSION}|{case_id}|{seed}|{split}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def draw_unit(case_id, seed, split, count, dimension):
    import numpy as np
    return np.random.default_rng(sampling_seed(case_id, seed, split)).random((count, dimension))


def observation_times(row, points):
    import numpy as np
    start, end = row['quality_audit']['support']['time_range']
    blocks = [start+(end-start)*draw_unit(row['case_id'], s, part, points, 1)[:, 0]
              for s in SEEDS for part in SPLITS]
    times = np.concatenate(blocks)
    assert np.all(times > start) and np.all(times < end)
    assert np.unique(times).size == times.size, 'Random observation-time collision; no silent resampling'
    return times


def evaluate(text, x):
    from .expression import evaluate_expression
    return evaluate_expression({'equation': 'fresh-seed-audit', 'analysis_expression': text}, x)


def validate_values(row, x, y, reference=None):
    import numpy as np
    from .expression import original_text
    assert x.dtype == y.dtype == np.float64
    assert np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite fresh samples; do not filter'
    assert x.shape == (len(y), row['variables'])
    assert np.all(x.std(axis=0) > 0) and y.std() > 0
    raw = original_text(row)
    checks = {}
    for name, other in [('original_expression', evaluate(raw, x) if raw else None),
                        ('source_reference', reference)]:
        if other is None: continue
        other = np.asarray(other, dtype=np.float64).reshape(-1)
        assert np.isfinite(other).all(), name
        scale = max(float(np.max(np.abs(y))), float(np.max(np.abs(other))), np.finfo(float).tiny)
        nmse = float(np.mean(((y-other)/scale)**2)/np.var(y/scale))
        assert nmse < 1e-10, (row['case_id'], name, nmse)
        checks[name+'_agreement_nmse'] = nmse
    guards = row['quality_audit'].get('original_syntax_guards', []) + row['quality_audit'].get('domain_guards', [])
    for guard in guards:
        z = np.asarray(evaluate(guard['expression'], x))
        rule = guard['rule']
        okay = {'positive': z > 0, 'nonnegative': z >= 0, 'nonzero': z != 0,
                'unit_interval': np.abs(z) <= 1}[rule]
        assert np.isfinite(z).all() and np.all(okay), (row['case_id'], guard)
    return dict(**checks, guards_checked=len(guards), points=len(y),
                x_min=x.min(0).tolist(), x_max=x.max(0).tolist(),
                y_min=float(y.min()), y_max=float(y.max()), y_std=float(y.std()))


def fetch_source(row, destination):
    """Fetch exactly pinned source files, never search for alternate models."""
    from .source_files import fetch_source as verified_fetch
    return verified_fetch(row, destination)


def bio_states(row, times, output):
    import numpy as np
    import roadrunner
    paths = fetch_source(row, output/'sources'/row['source_model'])
    support = row['quality_audit']['support']
    rr = roadrunner.RoadRunner(str(paths['sbml']))
    rr.conservedMoietyAnalysis = support['conserved_moiety_analysis']
    assert support['algorithm'] in ('KISAO:0000694', 'KISAO:0000019')
    rr.setIntegrator('cvode')
    for setting, value in support['applied_parameters'].items():
        setattr(rr.integrator, setting, value)
    rr.integrator.variable_step_size = False
    begin = float(row['sedml_simulation']['initialTime'])
    start, end = support['time_range']
    if begin < start: rr.simulate(begin, start, 2, selections=['time'])
    order = np.argsort(times)
    requested = np.concatenate(([start], times[order], [end]))
    selections = ['time']+row['input_selections']+[row['source_reaction']]
    matrix = np.asarray(rr.simulate(times=requested.tolist(), selections=selections), dtype=np.float64)
    assert matrix.shape == (len(times)+2, len(selections))
    assert np.array_equal(matrix[:, 0], requested), 'RoadRunner did not use requested observation times'
    restored = np.empty_like(matrix[1:-1]); restored[order] = matrix[1:-1]
    return restored[:, 1:-1].copy(), restored[:, -1].copy(), dict(
        source_sbml_sha256=digest(paths['sbml']), source_sedml_sha256=digest(paths['sedml']),
        solver='libRoadRunner '+roadrunner.__version__+' CVODE',
        settings=support['applied_parameters'], initial_state='unchanged source SBML',
        time_range=support['time_range'], sampling='independent uniform observation times; unchanged source horizon')


def math_states(row, times):
    import numpy as np
    from scipy.integrate import solve_ivp
    sys.path.insert(0, str(ROOT/'vendor/dysts-official-20260827/dysts-master'))
    from dysts import flows
    assert row['source_model'].startswith('dysts-flow:'), 'No discrete-map fallback'
    model = getattr(flows, row['source_model'].split(':', 1)[1])()
    start, end = row['quality_audit']['support']['time_range']
    assert start == 0
    order = np.argsort(times)
    ic = np.asarray(model.ic, dtype=np.float64).reshape(-1)
    rhs = lambda t, state: np.asarray(model.rhs(state, t), dtype=np.float64)
    jac = (lambda t, state: np.asarray(model.jac(state, t), dtype=np.float64)) if model.has_jacobian() else None
    sol = solve_ivp(rhs, (start, end), ic, method='Radau', t_eval=times[order],
                    first_step=model.dt, jac=jac, rtol=1e-10, atol=1e-10)
    assert sol.success and sol.y.shape == (len(ic), len(times)), sol.message
    states = np.empty_like(sol.y.T); states[order] = sol.y.T
    parameters = set(model.params)
    names = [n for n in inspect.signature(model._rhs).parameters if n != 't' and n not in parameters]
    assert len(names) == states.shape[1]
    columns = dict(zip(names, states.T)); columns['t'] = times
    x = np.column_stack([columns[n] for n in row['original_symbols']])
    try:
        actual = np.column_stack([np.broadcast_to(a, (len(times),)) for a in model.rhs(states, times)])
    except Exception:
        actual = np.asarray([model.rhs(s, t) for s, t in zip(states, times)])
    assert actual.shape == states.shape
    observed = actual[:, int(row['source_component'])-1]
    return x, observed, dict(solver='SciPy Radau, same source RHS/Jacobian and tolerances',
        rtol=1e-10, atol=1e-10, first_step=model.dt, initial_state=ic.tolist(),
        source_parameters=model.params, time_range=[start, end],
        source_base_sha256=digest(Path(inspect.getfile(type(model))).with_name('base.py')),
        source_flows_sha256=digest(inspect.getfile(type(model))),
        sampling='independent uniform observation times; unchanged audited Fourier horizon',
        solver_evaluations=sol.nfev)


def generate_row(args):
    row, output, points = args; output = Path(output)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    stamp = time.time(); key = row['audit_id'][:20]
    destination = output/'fresh_receipts'/f'{key}.json'
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get('status') == 'complete' and existing['version'] == VERSION:
            assert existing['points_per_split'] == points
            for record in existing['files'].values(): assert digest(output/record['file']) == record['sha256']
            return existing
    try:
        support = row['quality_audit']['support']; blocks = []
        times = None; reference = None
        if support['kind'] == 'box':
            bounds = np.asarray(support['bounds'], dtype=np.float64)
            assert bounds.shape == (row['variables'], 2) and np.all(bounds[:, 1] > bounds[:, 0])
            for seed in SEEDS:
                for part in SPLITS:
                    unit = draw_unit(row['case_id'], seed, part, points, row['variables'])
                    if support['law'] == 'LogUniform':
                        assert np.all(bounds > 0)
                        x = np.exp(np.log(bounds[:, 0])+unit*np.log(bounds[:, 1]/bounds[:, 0]))
                    else:
                        assert support['law'] == 'Uniform'
                        x = bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])
                    blocks.append(x)
            all_x = np.concatenate(blocks)
            details = dict(kind='box', law=support['law'], bounds=bounds.tolist())
        else:
            assert support['kind'] == 'trajectory'
            times = observation_times(row, points)
            if row['source_corpus'] == 'biochemistry': all_x, reference, details = bio_states(row, times, output)
            else: all_x, reference, details = math_states(row, times)
        all_x = np.asarray(all_x, dtype=np.float64)
        y = np.asarray(evaluate(row['analysis_expression'], all_x), dtype=np.float64).reshape(-1)
        check = validate_values(row, all_x, y, reference)
        files = {}; hashes = []; sequence = 0
        for seed in SEEDS:
            payload = {}; part_records = {}
            for part in SPLITS:
                lo, hi = sequence*points, (sequence+1)*points
                xpart, ypart = all_x[lo:hi], y[lo:hi, None]
                assert np.all(xpart.std(0) > 0) and ypart.std() > 0
                payload[f'x_{part}'] = xpart; payload[f'y_{part}'] = ypart
                payload[f'sample_ids_{part}'] = np.arange(lo, hi, dtype=np.int64)
                payload[f'sampling_seed_{part}'] = np.asarray(sampling_seed(row['case_id'], seed, part), dtype=np.uint64)
                if times is not None: payload[f'observation_times_{part}'] = times[lo:hi]
                fingerprint = array_hash(xpart); hashes.append(fingerprint)
                part_records[part] = dict(x_sha256=fingerprint, y_sha256=array_hash(ypart),
                    sampling_seed=sampling_seed(row['case_id'], seed, part), points=points)
                sequence += 1
            relative = f'seed_splits/{row["source_corpus"]}/{key}_seed{seed}.npz'
            path = output/relative; path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **payload)
            files[str(seed)] = dict(file=relative, sha256=digest(path), splits=part_records)
        assert len(set(hashes)) == 30, 'Repeated sample arrays'
        # Exact state repetition is reported, not silently filtered. Independently
        # sampled times can legitimately land on a deterministic steady state.
        unique_inputs = int(np.unique(all_x, axis=0).shape[0])
        report = dict(status='complete', version=VERSION, case_id=row['case_id'],
            audit_id=row['audit_id'], domain=row['source_corpus'], points_per_split=points,
            points_total=len(y), sampling_sets=30, distinct_sampling_streams=30,
            independent_fresh_draws=True, reused_old_pool=False, sample_ids_disjoint=True,
            unique_input_rows=unique_inputs, duplicate_input_rows=len(y)-unique_inputs,
            details=details, validation=check, files=files, elapsed_seconds=time.time()-stamp)
    except Exception as exc:
        report = dict(status='failed', version=VERSION, case_id=row['case_id'],
                      error=repr(exc), elapsed_seconds=time.time()-stamp)
    save_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--domains', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.dataset.read_text()) if r['source_corpus'] in args.domains]
    assert len(rows) == 20*len(args.domains)
    progress = args.output/'fresh_sampling_progress.json'
    completed = []; started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(generate_row, (row, args.output, 10000)) for row in rows]
        for future in as_completed(futures):
            report = future.result(); completed.append(report)
            save_json(progress, dict(version=VERSION, completed=len(completed), expected=len(rows),
                failed=sum(r['status'] != 'complete' for r in completed),
                elapsed_seconds=time.time()-started,
                cases=[{k:r.get(k) for k in ('case_id','status','elapsed_seconds','error')} for r in completed]))
            print(json.dumps({k:report.get(k) for k in ('case_id','status','points_total','elapsed_seconds','error')}), flush=True)
    assert all(r['status'] == 'complete' for r in completed), 'Fresh sampling incomplete; training must remain stopped'
    save_json(args.output/'fresh_sampling_complete.json', dict(version=VERSION, equations=len(rows),
        points_total=len(rows)*300000, sampling_sets=len(rows)*30,
        elapsed_seconds=time.time()-started, files={r['audit_id']:r for r in completed}))


if __name__ == '__main__': main()
