# Skeleton-KAN

**Transferring Interpretable Arithmetic Skeletons Across Scientific Equations**

Boyan Liu, School of Computer Science, University of Sheffield  
Contact: [BLiu61@sheffield.ac.uk](mailto:BLiu61@sheffield.ac.uk)

Skeleton-KAN constructs a finite bank of sparse sum/product neural architectures
from source equations, then trains their unary functions on a different target
domain. Construction does not access the target formula, observations or input
distribution. Target validation selects one frozen architecture per equation.

The code includes KS-inspired Interaction Envelope Synthesis (**KS-IES**, command
key `v2`) and three alternative skeleton constructors: TED, FGW and budgeted
ACUOS2, adapted through finite representatives to the same composition grammar.
Neural baselines are MLP, Fourier-MFN and MultKAN; a coordinate-connected SMPF
structure family provides an architecture ablation.

## Repository layout

```text
structured_kan/
  model/               StructuredKAN, KAN, GaussianBasis and neural baselines
  optimizer/           independent seed-batched L-BFGS and line search
  structure_builder/   source-only synthesis and external-method adapters
  dataset/             sampling, standardization, metrics and PDE definitions
  scripts/             construction, training and verification entry points
  tests/               component and pipeline regression tests
  vendor/              licensed PyKAN, dysts and official MFN reference code
  data/
    equation_set/      Domain300, eval80, held-out folds and PDE-10
    catalogues/        frozen skeleton banks and construction receipts
    train_config/      reproducible experiment configurations
    results/           ignored output directory for your runs
paper/scripts/         preparation and unfloored result-reporting helpers
setup_dependencies.py  restore remaining pinned external sources
verify_release.py      integrity and scientific-configuration audit
RESULTS.md             complete result archive and reporting convention
```

## Quick checks

Use Python 3.12 from the repository root:

```sh
python -B verify_release.py
python -m structured_kan.scripts.train --config mlp18_domain80.json --validate-only
```

The release verifier is standard-library-only: it checks hashes, Python syntax,
300 source records, 80 evaluation records, four disjoint held-out folds, 19
launch configurations and their frozen-bank references. It does not train or
download anything. Numerical component tests require their listed dependencies.

## Dependencies and platform support

The public release has **no Linux-only or remote-only execution assertion**.
Model components and configuration inspection can run on Windows, macOS or
Linux when their dependencies are available. Full research launch configurations
still request CUDA: removing an operating-system guard does not make a CUDA
benchmark a CPU benchmark. Linux/CUDA is the environment used for the paper's
results. Native Maude and Torch compilation availability depend on the platform.

PyKAN (upstream and optimized snapshots), dysts and the official MFN reference
are included with their original licenses. Restore the remaining native sources
and biological model files, then check their exact hashes:

```sh
python -B setup_dependencies.py
python -B setup_dependencies.py --verify-only
python -B verify_release.py --with-dependencies
```

Setup does not install Python packages, execute downloaded code or train models.
Repeated setup is idempotent; modified files and hash mismatches are rejected.
Use `--cache PATH --offline` with a populated cache for offline restoration.
See [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing dependencies.

Use separate environments for construction and fitting, because their pinned
SymPy versions differ. In the following commands, `python` means the activated
environment's interpreter. On Windows activate with `.venv-train\Scripts\Activate.ps1`;
on Linux/macOS use `source .venv-train/bin/activate` (similarly for `.venv-build`).

```sh
python -m venv .venv-build
# Activate .venv-build, then:
python -m pip install -r structured_kan/structure_builder/requirements-locked.txt

python -m venv .venv-train
# Activate .venv-train, then:
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r structured_kan/model/requirements-multkan.txt
python -m pip install networkx==3.3
```

The CUDA installation line reproduces the paper environment, not a universal
installation command. Choose the appropriate PyTorch wheel for your machine.
On platforms without a compatible Torch compilation backend, set
`"compile_mode": null` in a **copy** of the training configuration. This runs
eagerly without reducing the optimization budget; keep the frozen configurations
unchanged if checking release hashes. Full platform-specific performance and
accuracy replication is not implied by the portability changes.

## Rebuild a skeleton bank

In the construction environment:

```sh
python -m structured_kan.scripts.build_catalogue --method v2 --k 18 --workers 8 --threads 10 --heldout-domain symbolic_search --source structured_kan/data/equation_set/folds/heldout_symbolic_search/training225.json --output structured_kan/data/results/rebuilt_ksies/heldout_symbolic_search
```

Other held-out domains are `physics`, `mathematics` and `biochemistry`;
alternative method keys are `ted`, `fgw_sum` and `acuos2`. Each transfer bank
uses 225 source equations and excludes all 75 equations of its held-out domain.
For PDE construction use `--heldout-domain pde` and the full
`structured_kan/data/equation_set/domain300/selected/domain300_with_splits.json`.
Use a fresh output directory. Frozen banks are already supplied, so rebuilding
is optional for neural reproduction.

## Run the benchmarks

In the training environment:

```sh
python paper/scripts/prepare_paper_run.py --task transfer --name transfer
python -m structured_kan.scripts.train --config transfer_shard0.json --validate-only
python -m structured_kan.scripts.train --config transfer_shard0.json
python -m structured_kan.scripts.train --config transfer_shard1.json

python paper/scripts/prepare_paper_run.py --task pde --name pde
python -m structured_kan.scripts.pde_benchmark --config pde.json
```

The two Transfer shards can run sequentially on one GPU or on two separate
machines/GPUs. Each controller uses two worker processes. Do not run multiple
controllers on the same output directory. Existing completed results are resumed.
Each equation has ten independently sampled seeds, each with 10,000 training,
10,000 validation and 10,000 test points. Training uses train-only standardization,
50 outer L-BFGS calls, at most 20 inner iterations and independent strong-Wolfe
line searches. PDE training uses residuals and boundary/initial conditions,
not interior reference-solution labels.

Each full method requires 14,400 Transfer fits and 1,800 PDE fits. These are
full experiments, not quick tests. Baseline examples:

```sh
python -m structured_kan.scripts.train --config mlp18_domain80.json
python -m structured_kan.scripts.train --config ted_k18_dynamic_g_domain80.json
python -m structured_kan.scripts.pde_benchmark --config fgw_sum_pde10.json
```

## Results and reporting

All **12,960 configurations / 129,600 seed records** are provided as a downloadable
[release asset](https://github.com/188162600/Skeleton-KAN/releases/tag/v1.0.0),
rather than adding about 40,000 result files to every code checkout.
See [RESULTS.md](RESULTS.md) for the checksum and extraction instructions.

Select one configuration per equation by mean log validation NMSE across its
ten seeds, **after all 18 configurations finish**. Test scores are report-only.
Do not select a different architecture per seed. Main GMSE/GNMSE results have
no numerical floor. Recompute a report with:

```sh
python paper/scripts/report_paper_run.py --runs structured_kan/data/results/transfer_shard0 structured_kan/data/results/transfer_shard1 --expected-equations 80 --output structured_kan/data/results/transfer_report.json
```

For PDE use `--expected-equations 10`. Frozen source/evaluation records,
normalization rules, sampling protocols, optimization budgets and archived
numerical results are preserved. Source-pool discovery and the full development
history are not rerun: this release starts from the audited Domain300 records.

## License and citation

Original code is MIT licensed, copyright 2026 Boyan Liu. Third-party code and
external datasets keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Use [CITATION.cff](CITATION.cff) to cite this software. No publication DOI or
journal acceptance is asserted.
