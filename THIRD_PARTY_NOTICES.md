# Third-party sources and licenses

The root MIT license covers original project code, not all files in this
repository. Third-party copyright and license notices remain intact.

| Component | Pinned revision | Distribution and terms |
| --- | --- | --- |
| [PyKAN](https://github.com/KindXiaoming/pykan) | `ecde4ec3274d3bef1ad737479cf126aed38ab530` | Bundled upstream and optimized snapshots, MIT; copyright Ziming Liu. Each snapshot retains `LICENSE`. |
| [dysts](https://github.com/williamgilpin/dysts) | `2a03f1ae7b0680b0470458783dcb4664660e131a` | Bundled required source subset, Apache-2.0; original license and embedded notices retained. |
| [Multiplicative Filter Networks](https://github.com/boschresearch/multiplicative-filter-networks) | `58c3c4dd5908bce9d5c3b077926cb588cc6052da` | Optional bundled upstream reference for equivalence tests, AGPL-3.0; its source and full license are under `structured_kan/vendor/fourier_mfn_official/`. Not relicensed under MIT. |
| [Published FGW implementation](https://github.com/tvayer/FGW) | `3d2128a5e4e2cb8ed8a272295feff13641a36cef` | Downloaded during setup. The pinned snapshot has no standalone license; no redistribution permission is inferred. Original notices are preserved. |
| [ACUOS2](http://safe-tools.dsic.upv.es/acuos2/) | Archive SHA-256 `d9214e03f135dc0798d198bbfea48b4b4fdab6d0e51322b13d74b3b402d311e6` | Downloaded from the authors during setup; original notices retained, no relicensing. |
| [BioModels source cache](https://github.com/sys-bio/temp-biomodels) | `6a09daf46af1bb89e4857436b623a7b8720863ad` | SBML/SED-ML files downloaded during setup. Model-specific provenance is retained in equation records; the project's MIT license does not relicense these sources. |

`DEPENDENCIES.lock.json` records every upstream object and installed-file hash.
The optimized PyKAN snapshot contains the benchmark's local performance changes;
the lock file records its exact reconstruction edits relative to the upstream
revision. The upstream snapshot is supplied separately for comparison. No claim
is made that either is an unmodified reproduction of every original-paper run.

Package-installed dependencies (PyTorch, POT, Maude, SciPy and others) retain
their own licenses. Setup does not change these terms. The downloadable results
archive contains numerical records and provenance, not a new license grant over
the source datasets.

