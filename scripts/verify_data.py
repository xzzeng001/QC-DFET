"""Verify dataset integrity from the repository's SHA-256 manifest."""
from pathlib import Path
import hashlib
import sys

root = Path(__file__).resolve().parents[1]
data = root / 'data'
expected = set()
failures = []
for line in (data / 'checksums.sha256').read_text(encoding='utf-8').splitlines():
    digest, name = line.split('  ', 1)
    path = (data / name).resolve()
    if not path.is_relative_to(data.resolve()):
        raise ValueError(f'Path outside dataset: {name}')
    expected.add(name)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        failures.append(name)
if failures:
    print('FAILED: missing or modified files: ' + ', '.join(failures))
    sys.exit(1)
print(f'PASS: {len(expected)} dataset files match their SHA-256 checksums.')
extra = {p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()} - expected - {'checksums.sha256'}
if extra:
    print('Additional untracked/generated files: ' + ', '.join(sorted(extra)))
