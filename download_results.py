"""Download the complete public result archive and verify its SHA-256."""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request

URL = 'https://github.com/188162600/Skeleton-KAN/releases/download/v1.0.0/skeleton-kan-results-v1.0.0.zip'
SHA256 = '9e26ab0212f3e09e92fede0fe312020669d2ec8a4988e3f0d7155b3d5d22b46e'
BYTES = 56212586


def verify(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    if path.stat().st_size != BYTES or h.hexdigest() != SHA256:
        raise ValueError('Result archive size or SHA-256 mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results-download/skeleton-kan-results-v1.0.0.zip'))
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output)
        print('Existing archive verified:', args.output)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix='.download-', delete=False) as f:
            temp = Path(f.name)
            total = 0
            request = urllib.request.Request(URL, headers={'User-Agent': 'Skeleton-KAN/1.0'})
            with urllib.request.urlopen(request, timeout=120) as response:
                if not response.url.startswith('https://'):
                    raise ValueError('Refusing an HTTPS downgrade')
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > BYTES:
                        raise ValueError('Result archive exceeds the expected size')
                    f.write(chunk)
        verify(temp)
        os.link(temp, args.output)  # Fail rather than overwrite a concurrent file.
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print('Downloaded and verified:', args.output)
    print('Extract the archive, then run its verify_results.py from the extracted result root.')


if __name__ == '__main__':
    main()
