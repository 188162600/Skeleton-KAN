"""Verified, atomic downloads of the frozen BioModels source files."""
from __future__ import annotations
import hashlib
import http.client
import os
from pathlib import Path, PurePosixPath
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

COMMIT = '6a09daf46af1bb89e4857436b623a7b8720863ad'
CACHE = Path(__file__).resolve().parents[1]/'data/equation_set/sources/biochemistry'


def source_records(row):
    """Select the audited protocol by content hash, not the first candidate."""
    support = row['quality_audit']['support']
    expected = support.get('sedml_sha256', row['source_receipt']['sha256'])
    candidates = [row] + list(row.get('protocols', []))
    matches = [p for p in candidates if p.get('source_receipt', {}).get('sha256') == expected]
    if not matches:
        raise ValueError(f"No frozen SED-ML receipt matches support SHA for {row['case_id']}")
    protocol = matches[0]
    assert float(protocol['sedml_simulation']['initialTime']) == float(row['sedml_simulation']['initialTime'])
    assert float(protocol['sedml_simulation']['outputStartTime']) == support['time_range'][0]
    assert float(protocol['sedml_simulation']['outputEndTime']) == support['time_range'][1]
    assert row['sbml_sha256'] == support['source_file_sha256']
    sbml = PurePosixPath(str(row['sbml_source_path']).replace('\\', '/')).name
    return [('sbml', 'final/'+row['source_model']+'/'+sbml, row['sbml_sha256']),
            ('sedml', protocol['source_receipt']['upstream_path'], expected)]


def source_url(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not relative.startswith('final/'):
        raise ValueError('Source path must stay within the pinned final/ tree')
    return f'https://raw.githubusercontent.com/sys-bio/temp-biomodels/{COMMIT}/' + urllib.parse.quote(relative, safe='/')


def fetch_pinned(relative, expected, path, *, attempts=4, timeout=30):
    """Never accept incomplete, corrupt, or different-version source content."""
    path = Path(path)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Cached source SHA mismatch: {path}')
        return path
    url = source_url(relative)
    last_error = None
    cached = CACHE.joinpath(*PurePosixPath(relative).parts[1:])
    payload = None
    if cached.is_file():
        payload = cached.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Packaged source SHA mismatch: {cached}')
    for attempt in range(attempts):
        if payload is not None:
            break
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'StructuredKAN-pinned-source/1'})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'Downloaded source SHA mismatch: {relative}')
            break
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            last_error = error
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            last_error = error
        if attempt+1 == attempts:
            raise RuntimeError(f'Pinned source unavailable after {attempts} attempts: {relative}') from last_error
        time.sleep(min(2**attempt, 8))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+'.', suffix='.part', dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        # Concurrent writers have verified identical content before publishing.
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def fetch_source(row, destination):
    destination = Path(destination)
    return {kind: fetch_pinned(relative, expected, destination/PurePosixPath(relative).name)
            for kind, relative, expected in source_records(row)}
