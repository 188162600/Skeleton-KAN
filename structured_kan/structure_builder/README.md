# Source-side skeleton construction

This folder constructs skeleton banks from source equations.
`model/StructuredKANBuilder.py` has a separate responsibility: realizing
one frozen structure as Torch children, affine routes and incoming unary maps.

The proposed constructor is **KS-inspired Interaction Envelope Synthesis (KS-IES)**,
where KS denotes Kolmogorov superposition. This is the sum/product constructor
name, not a neural-model or kernel name. Its public command key is `ks_ies`;
`v2` remains a compatibility alias in frozen records and existing API calls.
Frozen JSON filenames, run identities and builder IDs are preserved.

| Constructor | Role within the framework | Command key |
| --- | --- | --- |
| **KS-inspired Interaction Envelope Synthesis (KS-IES)** | **Proposed constructor** | `ks_ies` |
| FGW clustering + finite representative | Alternative constructor | `fgw_sum` |
| TED average-linkage + finite medoid | Alternative constructor | `ted` |
| Budgeted ACUOS² + finite representative | Alternative constructor | `acuos2` |

All four are constructor options inside Skeleton-KAN. The three alternatives
adapt published methods to the same framework; they are not external neural
baselines. MLP, Fourier-MFN and MultKAN serve that separate comparison role.

| Entry/module | Responsibility |
| --- | --- |
| `build.py` | Source-only constructor API, finite representatives, distinct-K shortfall loop, group JSON |
| `proposed.py` | KS-inspired Interaction Envelope Synthesis (KS-IES; proposed), stable key `v2` |
| `synthesis.py` | KS-IES canonical classes, compatible envelope merging, source-only refinement |
| `incidence.py` | Source route incidence and unary-placement reconstruction |
| `shared_reserve.py` | KS-IES core capacities, shared reserve and source-derived sharing priors |
| `semantics.py` | Explicit declared inputs; every noninput symbol/compound expression is constant |
| `terms.py` | Common executable-edge projection, constant-padding removal and unary-arity normalization |
| `native_trees.py` | Original-expression ASTs, unit-cost ordered TED, average-linkage groups |
| `fgw.py`, `fgw_pot.py` | Source-initialized FGW graph clustering and modern POT barycenters |
| `acuos2.py`, `pair_cache.py` | Native Maude ACUOS² pair generalization, checks and bounded pair execution |
| `matching.py` | Total rooted M/U matching and separate minimum-mismatch coverage |
| `cardinality.py`, `cache.py` | Native-count shortfall steps; persistent exact preprocessing cache |
| `catalogue.py` | I/O for already explicit model specifications; not a clustering algorithm |

## Build a group

Use the designated remote machine. From the parent of `structured_kan/`:

```text
python -m structured_kan.scripts.build_catalogue --method ted \
  --source structured_kan/data/equation_set/folds/heldout_symbolic_search/training225.json \
  --heldout-domain symbolic_search --k 18 --workers 8 --threads 10 \
  --output structured_kan/data/results/new_ted_symbolic_search
```

Methods: `ks_ies`, `ted`, `fgw_sum`, `acuos2` (`v2` is a compatibility alias). Use a new output directory; old results
are never overwritten. The result is one **whole group** in `catalogue.json`,
with source SHA-256, constructor settings and per-count diagnostics. No target
equations or numerical fits are inputs. These APIs accept the frozen,
constant-corrected operator-variable specifications accompanying each source
expression; they do not regenerate the dataset or infer inputs from free symbols.

The isolated construction environment is specified by
`requirements-locked.txt`. Torch is not imported for source-side construction.
`scripts/rebuild_queue.py` creates that environment on the data disk, validates
it, and runs Symbolic regression, Physics, Mathematics and Biology/Chemistry folds.
`workers=8` parallelizes independent TED distances. The published FGW outer loop
and sequential within-cluster ACUOS² reductions retain their existing ordering;
the option does not claim eight concurrent FGW optimizers.

## Preserved algorithms and explicit adapters

- **KS-inspired Interaction Envelope Synthesis (KS-IES; proposed, key `v2`):** same frequency-weighted source objective and up to six refinement swaps,
  then assigned-source incidence, core capacity and shared reserve; reserve
  quantile 0.95, global capacity quantile 0.85, independent-prefix role prior.
- **TED:** original AST → ordered unit-cost APTED → average linkage → finite
  cluster medoid → common φ projection.
- **FGW:** original AST graph → categorical features → source-initialized graph
  clustering with POT 0.9.6.post1 unregularized FGW barycenters → nearest source
  representative → common φ projection. This preserves the established modern
  POT variant, not bitwise equivalence to the older numerical repository.
- **ACUOS²:** original ASTs → native generalizers within TED groups → verified
  finite source instance → common φ projection. A pair exceeding 30 seconds is
  skipped and the prior pattern retained. Skipped members remain in the dataset
  and are not falsely claimed to be generalized. This is an explicitly
  approximate n-way catalogue, not a complete n-way minimal-generalizer set.

For finite alternative constructors, start native count at K; if conversion gives K−s distinct
builders, advance the native count by s. If a larger result supplies enough,
fill the shortfall without duplicate padding. No held-out score chooses K or a
representative. Native Maude/FGW vendor files are byte-preserved; provenance
records list loader adaptations and the source code from which adapters moved.

## Model boundary

The analyser can attach **optional complexity annotations `c_phi`**, using
source functions aligned to each slot. These are auxiliary metadata, not the
definition or objective of the structural constructor, and not Gaussian basis
counts. A neural realization can ignore them entirely.

Dynamic per-map `G` is a separate parameter-saving policy. In the current
experiments it happens to consume the annotations via `G_phi = h(c_phi)`,
rounding to nearest `{3,9,15}` with lower ties. Uniform `G` or another policy
can use the same skeleton without these annotations. The connection is an
experimental heuristic, not a necessary coupling or an accuracy guarantee.

The compatibility entry point `complexity.allocate()` currently packages both
steps into the saved training specification: `mean_complexity` is the analyser
output; `G` and `phi_basis_by_label` are the Gaussian realization's derived
settings. These fields must not be described as equivalent concepts. This
clarification does not change archived files, numerical rules or running jobs.

`StructuredKANBuilder.build_finite_topology()` converts a finite, exact-arity
tree to explicit affine raw routes and incoming Gaussian maps. It accepts
per-map G, and an output φ exists **only** when `output_G` is explicitly given.
It rejects shared-capacity metadata instead of silently discarding it.

The new finite-topology bridge is tested, but complete migration of archived
training configurations still requires source-native sampling, per-map dynamic
budget allocation and the shared-reserve model adapter. The named archived
training configs continue to refuse execution until those pieces are verified;
constructor completion alone does not mean neural training has restarted.

## Verification

Remote tests cover constants, declared inputs named `pi`, unary sum/product
equivalence, arity, exact TED, source-only guards, distinct-count completion,
POT transport marginals/barycenters, native ACUOS² pairs, v2 source-to-reserve
construction, and the common v2 projector. Finite-model tests check postorder
budget labels, parameter counts and explicit output-map behavior.

Completed four-fold v2, TED and ACUOS² rebuilds reproduce all **216/216** archived
builder topologies (18 per method/fold), using identical source hashes. v2 raw
incidence, edge-route, sharing-prior and capacity-quantile records also match.
This checks relocation fidelity, not held-out accuracy. Remote receipts and rebuilt
group JSONs are under `data/results/constructor_remote_receipts/` and
`data/catalogues/rebuilt_20260908/`.
