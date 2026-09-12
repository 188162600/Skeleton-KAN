# Public-release checks

The public release preserves the archived model, optimizer, frozen scientific
inputs, skeleton banks and training budgets. Changes remove 15 explicit
Linux-only execution guards and add a spawned-process path for native ACUOS2
pair operations on platforms without Linux `fork`. The Linux path is retained.
Unix peak-memory reporting is optional when the `resource` module is unavailable.

Checks performed before publication:

- Integrity verification of the public source manifest and all 153 pinned
  dependency files, including offline restoration of setup-only sources.
- Configuration verification: 300 source equations, 80 evaluation equations,
  four held-out folds, 19 launch configurations and 20 referenced skeleton banks.
- 45 CPU tests passed on Windows with Python 3.13 and PyTorch 2.11. These cover
  core components, topology specifications, route/map sharing, MLP, Fourier-MFN,
  MultKAN and platform handling. The MultKAN tests compare the optimized and
  upstream PyKAN forward values, gradients and serialized state.
- A scan of Git-visible files found no workstation paths, remote experiment
  endpoints, private keys or recognized credential patterns. This is a scoped
  release audit, not a guarantee against every possible secret format.

No optimizer-fitting or full benchmark training was performed for these release
checks. Cross-platform integrity/portability checks run in GitHub Actions;
they do not establish CUDA training equivalence on every platform. In
particular, native Maude availability and compiled Torch execution require
platform-appropriate dependencies. See the README for setup and launch commands.
