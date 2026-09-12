"""Read-only release checks; standard library only, with no training or network."""
import ast
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'structured_kan'
sys.path.insert(0, str(ROOT))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-dependencies', action='store_true')
    args = parser.parse_args()
    if args.with_dependencies:
        import setup_dependencies
        setup_dependencies.install(verify_only=True)
    manifest = read(ROOT / 'SOURCE_SHA256.json')
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve()), name
        assert sha(path) == expected, f'File changed: {name}'
        if name.endswith('.py'): ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
    audit = read(ROOT / 'REPRODUCIBILITY.json')
    parent_path = PROJECT / 'data/equation_set/domain300/selected/domain300_with_splits.json'
    subset_path = PROJECT / 'data/equation_set/eval80/subset80_with_splits.json'
    parent, subset = read(parent_path), read(subset_path)
    assert len(parent) == 300 and len(subset) == 80
    assert set(Counter(r['source_corpus'] for r in parent).values()) == {75}
    assert set(Counter(r['source_corpus'] for r in subset).values()) == {20}
    assert len({r['case_id'] for r in parent}) == 300
    records = {r['case_id']: r for r in parent}
    for row in subset:
        for key in ('source_corpus', 'variables', 'analysis_expression', 'operator_variable_spec', 'quality_audit'):
            assert row[key] == records[row['case_id']][key], (row['case_id'], key)
    assert all(2 <= r['variables'] <= 10 for r in parent)
    for folder in (PROJECT / 'data/equation_set/folds').iterdir():
        train = read(folder / 'training225.json')
        held = read(folder / 'heldout75.json')
        assert len(train) == 225 and len(held) == 75
        domain = folder.name.removeprefix('heldout_')
        assert all(r['source_corpus'] != domain for r in train)
        assert all(r['source_corpus'] == domain for r in held)
        assert not {r['case_id'] for r in train} & {r['case_id'] for r in held}

    def ref(item):
        path = PROJECT / item['path']
        assert sha(path) == item['sha256'], item['path']
        return read(path)
    groups = set()
    for name in audit['transfer_configurations'] + audit['pde_configurations']:
        c = read(PROJECT / 'data/train_config' / name)
        assert c['seeds'] == list(range(421, 431))
        assert (c['outer_steps'], c['max_inner_iterations'], c['history_size']) == (50, 20, 100)
        assert c['points_per_split'] == 10000 and c['workers'] in (1, 2), name
        if name in audit['transfer_configurations']: assert c['workers'] == 2, name
        for key in ('parent', 'subset', 'source', 'problems', 'template'):
            if key in c: ref(c[key])
        for domain, item in c.get('catalogues', {}).items():
            group = ref(item); groups.add(item['path'])
            source_path = PROJECT / item['source']
            assert sha(source_path) == item['source_sha256'] == group['training_sha256']
            assert len(read(source_path)) == group['training_equations'] == 225
            assert len(group['builders']) == group['builder_count'] == 18
            assert group['heldout_domain'] == domain and group['source_only']
        if c.get('construction'):
            construction_path = PROJECT / c['construction']
            base = read(construction_path)
            group = read(PROJECT / c['catalogue']); groups.add(c['catalogue'])
            assert base['training_sha256'] == sha(parent_path)
            assert base['training_equations'] == 300 and base['builder_count'] == 18
            assert base['construction']['existing_catalogue_used'] is False
            assert group['complexity_allocation']['base_catalogue_sha256'] == sha(construction_path)
            assert group['complexity_allocation']['source_rows'] == 300
            assert group['complexity_allocation']['heldout_rows_read'] == 0
            if 'construction_sha256' in c:
                assert c['construction_sha256'] == sha(construction_path)
                assert c['catalogue_sha256'] == sha(PROJECT / c['catalogue'])
        if 'architectures' in c:
            assert len(c['architectures']) == 18
            assert len({a['id'] for a in c['architectures']}) == 18

    # Bypass dataset/__init__: it eagerly imports numerical dependencies.
    spec = importlib.util.spec_from_file_location('release_source_files', PROJECT / 'dataset/source_files.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source_records, CACHE = module.source_records, module.CACHE
    cached = 0
    for row in subset:
        if row['source_corpus'] != 'biochemistry': continue
        for _, upstream, expected in source_records(row):
            path = CACHE / upstream.removeprefix('final/')
            if args.with_dependencies:
                assert sha(path) == expected, row['case_id']
                cached += 1
    print(json.dumps(dict(status='passed', files=len(manifest), equations=300,
        evaluation_equations=80, heldout_folds=4, launch_configurations=19,
        referenced_skeleton_banks=len(groups), cached_upstream_files_checked=cached,
        training_launched=False, network_access=False, dependencies_checked=args.with_dependencies), indent=2))


if __name__ == '__main__': main()
