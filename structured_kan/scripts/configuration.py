"""Resolve one named training JSON and validate its archived data references.

Standard library only: reading a configuration never starts a model or sampler.
"""
import hashlib
import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1]/'data'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_config(name, *, data_root=DATA):
    """Accept an explicit path or a unique filename under data/train_config."""
    direct = Path(name)
    if direct.is_file():
        return direct.resolve()
    if direct.name != str(name) or direct.suffix != '.json':
        raise FileNotFoundError(name)
    matches = list((Path(data_root)/'train_config').rglob(direct.name))
    if len(matches) != 1:
        raise ValueError(f'Expected one config named {name!r}; found {len(matches)}')
    return matches[0].resolve()


def inspect_config(name, *, data_root=DATA, verify_splits=False):
    config_path = resolve_config(name,data_root=data_root)
    config = load_json(config_path)
    if config.get('schema') != 'structured-kan.train-config.v1':
        raise ValueError('Unsupported training configuration schema')
    def referred(path, parent=config_path):
        return (parent.parent/path).resolve()
    catalogue_path = referred(config['catalogue'])
    dataset_path = referred(config['dataset']['records'])
    subset_path = referred(config['dataset']['subset'])
    for path, expected in ((catalogue_path,config['catalogue_sha256']),
                           (dataset_path,config['dataset']['records_sha256']),
                           (subset_path,config['dataset']['subset_sha256'])):
        if sha256(path) != expected:
            raise ValueError(f'SHA256 mismatch: {path}')
    group = load_json(catalogue_path)
    if group.get('schema') != 'structured-kan.catalogue-group.v1':
        raise ValueError('Expected one whole catalogue group JSON')
    builders = group['builders']
    if len(builders) != group['builder_count'] or len({b['builder'] for b in builders}) != len(builders):
        raise ValueError('Invalid catalogue size or duplicate builder IDs')
    original = referred(group['provenance'],catalogue_path)/'builders.jsonl'
    if sha256(original) != group['original_builders_sha256']:
        raise ValueError('Original catalogue hash mismatch')
    original_records = [json.loads(line) for line in original.read_text(encoding='utf-8').splitlines() if line.strip()]
    if original_records != builders:
        raise ValueError('Bundling changed constructor records')
    training_path = referred(group['training_data'],catalogue_path)
    if sha256(training_path) != group['training_sha256']:
        raise ValueError('Source training fold hash mismatch')
    training, subset, dataset = load_json(training_path),load_json(subset_path),load_json(dataset_path)
    domain = config['dataset']['heldout_domain']
    if domain != group['heldout_domain']:
        raise ValueError('Training configuration and catalogue held-out domains differ')
    identity = lambda r:(r['source_corpus'],r['case_id'])
    train_ids, eval_ids, dataset_ids = (set(map(identity,r)) for r in (training,subset,dataset))
    if len(training) != 225 or len(subset) != config['dataset']['evaluation_equations'] or len(dataset) != 300:
        raise ValueError('Dataset cardinality mismatch')
    if train_ids & eval_ids or not train_ids | eval_ids <= dataset_ids:
        raise ValueError('Fold overlap or equations missing from parent dataset')
    if any(r['source_corpus']==domain for r in training) or any(r['source_corpus']!=domain for r in subset):
        raise ValueError('Leave-one-domain-out split violation')
    if verify_splits:
        index_path = referred(config['dataset']['frozen_audit_splits'])
        for item in load_json(index_path):
            if sha256(index_path.parent/item['file']) != item['sha256']:
                raise ValueError(f"Split hash mismatch: {item['case_id']}")
    return dict(config=config,catalogue=group,config_path=str(config_path),
                summary=dict(name=config['name'],method=group['method'],builders=len(builders),
                    parent_equations=len(dataset),training_equations=len(training),
                    evaluation_equations=len(subset),heldout_domain=domain,
                    output_directory=str(referred(config['output_directory'])),
                    references_verified=True,execution=config['execution']['status']))
