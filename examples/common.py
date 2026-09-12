"""Small helpers shared by the two ordinary-closure L-BFGS examples."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from structured_kan.model.StructuredKANBuilder import StructuredKANBuilder
from structured_kan.model.initialization import perturb_trainable_
from structured_kan.model.shared_affine_spec import private_affine_shared_phi_spec
from structured_kan.optimizer.lbfgs import LBFGS


ROOT = Path(__file__).resolve().parents[1]
# The filename is a frozen compatibility identifier; the constructor is KS-IES.
BANK = ROOT / 'structured_kan/data/catalogues/pde10/v2_source300_k18_dynamic_g.json'


def arguments(description, name, default_builder='IAPB13', default_points=1024):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=421)
    parser.add_argument('--builder', default=default_builder)
    parser.add_argument('--points', type=int, default=default_points,
                        help='points per regression split / interior collocation set')
    parser.add_argument('--outer-steps', type=int, default=50)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'structured_kan/data/results' / name)
    args = parser.parse_args()
    if args.points < 4 or args.outer_steps < 1:
        parser.error('require at least 4 points and one outer step')
    return args


def standardization(values):
    mean = values.mean(0, keepdim=True)
    scale = values.std(0, correction=0, keepdim=True)
    if not bool(torch.isfinite(values).all()) or bool((scale <= 0).any()):
        raise ValueError('normalization requires finite, nonconstant training values')
    return mean, scale


def build_model(args, normalized_training_inputs, *, bank_path=BANK, route_budgets=None):
    raw = bank_path.read_bytes()
    bank = json.loads(raw)
    if not bank['source_only'] or bank['training_equations'] not in (225, 300):
        raise ValueError('expected a frozen source-only KS-IES skeleton bank')
    records = {item['builder']: item for item in bank['builders']}
    if args.builder not in records:
        raise ValueError(f'unknown builder {args.builder}; choose from {list(records)}')
    record = copy.deepcopy(records[args.builder])
    if route_budgets is not None:
        record['private_phi_G_by_route'] = route_budgets
    spec, metadata = private_affine_shared_phi_spec(
        record, normalized_training_inputs.shape[1])
    model = StructuredKANBuilder(normalized_training_inputs.shape[1],
        dtype=torch.float64, device=args.device).build(
            spec, train_inputs=normalized_training_inputs)
    perturb_trainable_(model, args.seed)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert actual == metadata['expected_parameters'], (actual, metadata)
    metadata.update(constructor='KS-IES', builder=args.builder,
                    bank_sha256=hashlib.sha256(raw).hexdigest(), parameters=actual)
    return model, spec, metadata


def benchmark_equation():
    reference = json.loads((ROOT/'examples/benchmark_equation.json').read_text())
    rows = json.loads((ROOT/'structured_kan/data/equation_set/eval80/subset80_with_splits.json').read_text())
    row = next(r for r in rows if r['case_id'] == reference['case_id'])
    assert row['expression'] == reference['expression']
    assert row['quality_audit']['support']['law'] == 'Uniform'
    for name in ('catalogue', 'source'):
        assert hashlib.sha256((ROOT/reference[name]).read_bytes()).hexdigest() == reference[name+'_sha256']
    sources = json.loads((ROOT/reference['source']).read_text())
    assert len(sources) == 225 and all(r['source_corpus'] != 'mathematics' for r in sources)
    return row, reference


def make_optimizer(model):
    # This is the project's standalone optimizer, not torch.optim.LBFGS.
    return LBFGS(model.parameters(), lr=1., max_iter=20, history_size=100,
                 tolerance_grad=1e-32, tolerance_change=1e-32,
                 tolerance_ys=1e-32, line_search_fn='strong_wolfe',
                 two_loop_mode='nosync_fma', scalar_mode='coalesced_host')


def fit(model, optimizer, closure, validation_loss, outer_steps):
    best_loss = float(validation_loss())
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    for step in range(1, outer_steps + 1):
        optimizer.step(closure)
        loss = float(validation_loss())
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError('non-finite validation loss')
        if loss < best_loss:
            best_loss, best_state, best_step = loss, copy.deepcopy(model.state_dict()), step
        if step == 1 or step % 10 == 0 or step == outer_steps:
            print(json.dumps(dict(outer_step=step, validation_nmse=loss)), flush=True)
    model.load_state_dict(best_state)
    return best_step


def metrics(prediction, target):
    mse = (prediction - target).square().mean()
    variance = target.var(correction=0)
    if not bool(torch.isfinite(mse)) or not bool(variance > 0):
        raise FloatingPointError('metrics require finite error and positive target variance')
    return dict(mse=float(mse), nmse=float(mse / variance))


def save(args, model, spec, metadata, normalizers, result):
    result.update(seed=args.seed, device=str(args.device), **metadata,
                  optimizer='project standalone LBFGS', seed_batch=False,
                  outer_steps=args.outer_steps, max_inner_iterations=20)
    # The examples require a new output path; never overwrite another run.
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = dict(schema='skeleton-kan.single-model-example.v1',
        spec=spec, state_dict=model.state_dict(), normalizers=normalizers, result=result)
    torch.save(checkpoint, args.output / 'model.pt')
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


def check_output(args):
    if args.output.exists():
        raise FileExistsError(f'use a new --output directory: {args.output}')
