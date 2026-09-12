"""Unfloored, complete-equation reporting from saved seed-level results."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def log_mean(values):
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    return float('-inf') if 0 in values else math.fsum(map(math.log, values)) / len(values)


def gm(values):
    value = log_mean(values)
    return 0. if value == float('-inf') else math.exp(value)


def report(runs, expected):
    groups = defaultdict(dict)
    methods = set()
    for run in runs:
        for path in sorted((run / 'jobs').glob('*/*/result.json')):
            record = json.loads(path.read_text())
            if record.get('status') != 'complete':
                continue
            method = record['method']
            methods.add(method)
            key = (record.get('domain', 'pde'), record.get('case_id', record['equation']))
            builder = record['builder']
            if builder in groups[key]:
                raise ValueError(f'Duplicate equation/builder across runs: {key}, {builder}')
            seeds = record['seeds']
            assert len(seeds) == 10 and sorted(s['seed'] for s in seeds) == list(range(421, 431)), path
            assert all(s['method'] == method and s['builder'] == builder for s in seeds), path
            for seed in seeds:
                for split in ('validation', 'test'):
                    for metric in ('mse', 'nmse'):
                        value = seed['metrics'][split][metric]
                        assert math.isfinite(value) and value >= 0, (path, split, metric)
            groups[key][builder] = record
    if len(methods) != 1:
        raise ValueError(f'Expected exactly one method with results; got {methods}')
    selected, incomplete = [], []
    for (domain, case), candidates in sorted(groups.items()):
        if len(candidates) != 18:
            incomplete.append(dict(domain=domain, case_id=case, completed_builders=len(candidates)))
            continue
        # Shards for one equation must share a configuration. Banks may differ
        # across domains, but not across builders of the same equation.
        for field in ('configuration_sha256', 'catalogue_sha256'):
            hashes = {r[field] for r in candidates.values() if field in r}
            if len(hashes) > 1:
                raise ValueError(f'Inconsistent {field}: {domain}, {case}')
        def score(r):
            return (log_mean([s['metrics']['validation']['nmse'] for s in r['seeds']]), r['builder'])
        best = min(candidates.values(), key=score)
        row = dict(domain=domain, case_id=case, equation=best['equation'], builder=best['builder'],
                   parameters=best['parameters'],
                   test_gnmse=gm([s['metrics']['test']['nmse'] for s in best['seeds']]),
                   test_gmse=gm([s['metrics']['test']['mse'] for s in best['seeds']]))
        selected.append(row)
    def aggregate(rows):
        return dict(equations=len(rows), test_gnmse=gm([r['test_gnmse'] for r in rows]),
                    test_gmse=gm([r['test_gmse'] for r in rows]),
                    mean_selected_parameters=math.fsum(r['parameters'] for r in rows) / len(rows))
    result = dict(method=next(iter(methods)), report_floor=None,
                  selection='one builder per equation, unfloored mean log validation NMSE over seeds421-430; builder-ID tie',
                  expected_equations=expected, complete_equations=len(selected),
                  incomplete_equations=incomplete, selected=selected)
    if selected:
        result['overall'] = aggregate(selected)
        result['by_domain'] = {d: aggregate([r for r in selected if r['domain'] == d]) for d in sorted({r['domain'] for r in selected})}
    result['complete'] = len(selected) == expected and not incomplete
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--expected-equations', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    result = report(args.runs, args.expected_equations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ('selected', 'incomplete_equations')}, indent=2))
    if not result['complete'] and not args.allow_partial:
        raise SystemExit('Incomplete candidate sweeps: result is diagnostic only, not the requested full benchmark')


if __name__ == '__main__':
    main()
