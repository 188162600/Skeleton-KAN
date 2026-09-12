"""Restore pinned external sources without running downloaded code (stdlib only)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
MAX_DOWNLOAD = 64 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '\\' in name or ':' in name
            or any(x in ('', '.', '..') for x in name.split('/'))):
        raise ValueError(f'Unsafe relative path: {name}')
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes installation root: {name}')
    current = root
    for part in p.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Refusing symbolic link: {name}')
    return result


def require_hash(data, expected, label):
    if digest(data) != expected:
        raise ValueError(f'SHA-256 mismatch: {label}')


def atomic_new(path, data):
    """Publish only complete content; never replace an existing different file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite modified file: {path.name}')
        return
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.setup-', delete=False) as f:
            temp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Hard-link publication is atomic and fails if another writer got here first.
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f'Concurrent file conflict: {path.name}')
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def download(source, cache, offline=False):
    path = cache / source['sha256']
    if path.exists():
        data = path.read_bytes()
        require_hash(data, source['sha256'], 'cached download')
        return data
    if offline:
        raise ValueError(f'Missing offline cache object: {source["id"]}')
    url = source['url']
    if not (url.startswith('https://') or
            url == 'http://safe-tools.dsic.upv.es/acuos2/download/source-benchmarks.zip'):
        raise ValueError('Only pinned HTTPS sources and the hash-pinned official ACUOS2 archive are allowed')
    last = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'skeleton-review-setup/1'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if url.startswith('https://') and not response.url.startswith('https://'):
                    raise ValueError('Refusing an HTTPS-to-HTTP downgrade')
                data = response.read(MAX_DOWNLOAD + 1)
            if len(data) > MAX_DOWNLOAD:
                raise ValueError('Download exceeds size limit')
            require_hash(data, source['sha256'], source['id'])
            atomic_new(path, data)
            return data
        except (OSError, TimeoutError) as error:
            last = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Download failed: {source["id"]}') from last


def restore_file(recipe, blob):
    if 'member' in recipe:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            name = recipe['member']
            # No extractall: only explicitly named, hash-checked members are read.
            safe_path(Path(tempfile.gettempdir()), name)
            entries = [i for i in archive.infolist() if i.filename == name]
            if len(entries) != 1:
                raise ValueError(f'Missing or duplicate archive member: {name}')
            info = entries[0]
            if stat.S_ISLNK(info.external_attr >> 16) or info.is_dir() or info.file_size > MAX_DOWNLOAD:
                raise ValueError(f'Invalid archive member: {name}')
            blob = archive.read(info)
    require_hash(blob, recipe['upstream_sha256'], recipe['path'] + ' upstream')
    if 'edits' in recipe:
        lines = blob.decode('utf-8').replace('\r\n', '\n').splitlines(keepends=True)
        result, cursor = [], 0
        for start, end, replacement in recipe['edits']:
            if not (cursor <= start <= end <= len(lines)):
                raise ValueError('Invalid patch offset')
            result.extend(lines[cursor:start])
            result.extend(replacement)
            cursor = end
        result.extend(lines[cursor:])
        text = ''.join(result)
        if recipe['newline'] == 'crlf':
            text = text.replace('\n', '\r\n')
        elif recipe['newline'] != 'lf':
            raise ValueError('Unknown newline convention')
        blob = text.encode('utf-8')
    require_hash(blob, recipe['sha256'], recipe['path'] + ' installed')
    return blob


def verify_manifest(root):
    manifest = json.loads((root / 'SOURCE_SHA256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        require_hash(safe_path(root, name).read_bytes(), expected, name)


def install(root=ROOT, *, verify_only=False, offline=False, cache=None, workers=6):
    root = Path(root).resolve()
    verify_manifest(root)
    lock = json.loads((root / 'DEPENDENCIES.lock.json').read_text(encoding='utf-8'))
    recipes = lock['files']
    names = [r['path'] for r in recipes]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate installation path')
    missing = []
    for recipe in recipes:
        path = safe_path(root, recipe['path'])
        if path.exists():
            require_hash(path.read_bytes(), recipe['sha256'], recipe['path'])
        else:
            missing.append(recipe)
    if verify_only:
        if missing:
            raise ValueError(f'{len(missing)} dependencies missing; run python setup_dependencies.py')
        return dict(status='passed', files_verified=len(recipes), files_installed=0, network_used=False)
    cache = Path(cache).resolve() if cache else root / '.dependency-cache'
    cache.mkdir(parents=True, exist_ok=True)
    wanted = {r['source'] for r in missing}
    sources = {s['id']: s for s in lock['sources']}
    if not wanted <= sources.keys():
        raise ValueError('Unknown download reference')
    blobs = {}
    def fetch(key):
        data = download(sources[key], cache, offline)
        print(f'Verified download: {key}', flush=True)
        return key, data
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, data in pool.map(fetch, sorted(wanted)):
            blobs[key] = data
    # Validate and stage EVERYTHING before adding any dependency to the project.
    with tempfile.TemporaryDirectory(prefix='.dependency-stage-', dir=root) as staging:
        stage = Path(staging)
        for recipe in missing:
            data = restore_file(recipe, blobs[recipe['source']])
            target = safe_path(stage, recipe['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # Install notices first; don't publish code without its downloaded notices.
        ordered = sorted(missing, key=lambda r: (not any(x in Path(r['path']).name.lower()
                         for x in ('license', 'copying', 'notice', 'readme')), r['path']))
        for recipe in ordered:
            atomic_new(safe_path(root, recipe['path']), safe_path(stage, recipe['path']).read_bytes())
    for recipe in recipes:
        require_hash(safe_path(root, recipe['path']).read_bytes(), recipe['sha256'], recipe['path'])
    return dict(status='passed', files_verified=len(recipes), files_installed=len(missing),
                sources_required=len(wanted), offline=offline, training_launched=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Require an already verified download cache')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    print(json.dumps(install(verify_only=args.verify_only, offline=args.offline,
                            cache=args.cache, workers=args.workers), indent=2))


if __name__ == '__main__':
    main()
