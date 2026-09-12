"""Content-addressed exact preprocessing cache, shared across native counts.

Fold vocabularies remain source-only. Cached objects never change native inputs.
Disabled unless NATIVE_PREPROCESS_CACHE names a data-disk SQLite file.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

_CONNECTION = None
_IDENTITY = None
_VERSIONS = {}
STATS = {'hit': 0, 'miss': 0, 'write': 0}


def fingerprint(*paths):
    key = tuple(str(x) for x in paths)
    if key not in _VERSIONS:
        h = hashlib.sha256()
        for path in paths:
            h.update(Path(path).read_bytes())
        _VERSIONS[key] = h.hexdigest()
    return _VERSIONS[key]


def connection():
    global _CONNECTION, _IDENTITY
    path = os.environ.get('NATIVE_PREPROCESS_CACHE')
    if not path:
        return None
    identity = (os.getpid(), path)
    if identity != _IDENTITY:
        if _CONNECTION is not None:
            _CONNECTION.close()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _CONNECTION = sqlite3.connect(path, timeout=120)
        _CONNECTION.execute('PRAGMA journal_mode=WAL')
        _CONNECTION.execute('PRAGMA synchronous=NORMAL')
        _CONNECTION.execute('CREATE TABLE IF NOT EXISTS cache (namespace TEXT, key TEXT, payload TEXT, created REAL, PRIMARY KEY(namespace,key))')
        _CONNECTION.commit()
        _IDENTITY = identity
    return _CONNECTION


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def get(namespace, identity):
    db = connection()
    if db is None:
        return None
    row = db.execute('SELECT payload FROM cache WHERE namespace=? AND key=?', (namespace, key(identity))).fetchone()
    STATS['hit' if row else 'miss'] += 1
    return json.loads(row[0]) if row else None


def put(namespace, identity, value):
    db = connection()
    if db is None:
        return
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    db.execute('INSERT OR IGNORE INTO cache VALUES (?,?,?,?)', (namespace, key(identity), payload, time.time()))
    db.commit()
    STATS['write'] += 1


def put_many(namespace, items):
    db = connection()
    if db is None:
        return
    now = time.time()
    db.executemany('INSERT OR IGNORE INTO cache VALUES (?,?,?,?)',
        ((namespace, key(identity), json.dumps(value, allow_nan=False), now) for identity,value in items))
    db.commit()


def tree_from_payload(payload):
    from .native_trees import ExprTree
    return ExprTree(payload['label'], [tree_from_payload(c) for c in payload['children']])


def tree_identity(tree):
    return tree.signature()


def distance_identity(a, b):
    # Unit-cost ordered TED is symmetric; do not alter either child order.
    return sorted([a.signature(), b.signature()])
