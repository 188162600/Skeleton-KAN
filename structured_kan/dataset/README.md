# Frozen dataset execution

The training configuration identifies `data/equation_set/eval80/subset80_with_splits.json`
by SHA-256. Keep that file, the parent dataset and its source support unchanged.

Each equation has ten independent seeded train/validation/test triples: seeds
421–430, 10,000 points per split, 30 sampling streams and 300,000 points total.
Sampling and scientific validation run on the remote host, not on the client.
Only training statistics standardize the three splits.

Biology/Chemistry uses joint source-model trajectories on the frozen SED-ML
time interval. It does not sample species independently or change initial
conditions. `source_files.py` selects the protocol whose SHA matches
`quality_audit.support.sedml_sha256`; the first listed protocol is not necessarily
the audited one. Compound constants, equations, domains and solver settings are
not changed by source retrieval.

Pinned source XML is packaged under
`data/equation_set/sources/biochemistry/`, with a provenance manifest. These are
source files, not sampled data. Every file is verified against the frozen
dataset SHA before use. If no packaged file is available, the loader URL-escapes
the pinned upstream path and retries transient network failures. Publication to
the cache is atomic. A wrong hash, permanent HTTP error, or invalid data fails
explicitly; there is no silent equation replacement, filtering or range change.

Remote full-size preparation and verification:

```bash
python -m unittest structured_kan.tests.test_source_files
python -m structured_kan.scripts.audit_biology_data \
  --project /path/to/structured_kan \
  --config /path/to/structured_kan/data/train_config/ted_k18_dynamic_g_domain80.json \
  --audit /data-disk/dataset_audit --workers 4
```

This verifies all 20 biology cases and every other already-prepared evaluation
case with the production loader. Reports retain source hashes, expression vs
SBML reaction agreement, split fingerprints, finite-value checks, train-only
normalization checks and any floating-point centering residual. A provenance
replay checks any old SED-ML receipt discrepancy without rewriting old samples.

Resume the original training command only after `data_verified.json` succeeds.
The existing queue skips complete results; do not remove results or checkpoints.
