"""Prepare fresh paper-run JSON files without sampling or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'structured_kan'
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('transfer', 'pde'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--catalogue-root', type=Path,
                        help='Fresh four-fold construction root; Transfer only')
    parser.add_argument('--validate-only', action='store_true',
                        help='Inspect all inputs and intended outputs; write nothing')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name):
        parser.error('Name must be a simple new run identifier')
    if args.catalogue_root and args.task != 'transfer':
        parser.error('--catalogue-root is supported for Transfer only')
    templates = ([f'v2_private_affine_shared_phi_dynamic_g_domain80_shard{i}.json' for i in (0, 1)]
                 if args.task == 'transfer' else ['v2_private_affine_shared_phi_pde10.json'])
    pending = []
    for i, template in enumerate(templates):
        path = PROJECT / 'data/train_config' / template
        if args.task == 'transfer':
            from structured_kan.scripts.benchmark import inspect_config
        else:
            from structured_kan.scripts.pde_benchmark import inspect_config
        _, original, _ = inspect_config(path)
        config = copy.deepcopy(original)
        name = f'{args.name}_shard{i}' if args.task == 'transfer' else args.name
        destination = PROJECT / 'data/train_config' / f'{name}.json'
        output = PROJECT / 'data/results' / name
        if destination.exists() or output.exists():
            raise FileExistsError(f'Use a fresh name; refusing to overwrite {destination} or {output}')
        config['output'] = output.relative_to(PROJECT).as_posix()
        if args.catalogue_root:
            for domain, entry in config['catalogues'].items():
                bank_path = (args.catalogue_root / f'heldout_{domain}/catalogue.json').resolve()
                bank = json.loads(bank_path.read_text())
                source = PROJECT / entry['source']
                rows = json.loads(source.read_text())
                assert bank['method'] == 'v2' and bank['heldout_domain'] == domain
                assert bank['source_only'] and bank['builder_count'] == len(bank['builders']) == 18
                assert bank['training_equations'] == len(rows) == 225
                assert all(row['source_corpus'] != domain for row in rows)
                assert bank['training_sha256'] == entry['source_sha256'] == sha(source)
                # Keep paths portable; custom banks must travel with the checkout.
                entry['path'] = bank_path.relative_to(PROJECT.resolve()).as_posix()
                entry['sha256'] = sha(bank_path)
        pending.append((destination, config))
    for destination, config in pending:
        if not args.validate_only:
            with destination.open('x', encoding='utf-8') as stream:
                json.dump(config, stream, indent=2)
            inspect_config(destination)
        print(json.dumps(dict(config=str(destination), output=config['output'],
                              task=args.task, validation_only=args.validate_only)))


if __name__ == '__main__':
    main()
