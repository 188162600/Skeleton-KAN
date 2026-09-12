# Directional third-order pilot

Experimental, safeguarded Chebyshev-style correction for small neural models.
This is not a claim of global convergence or a complete regularized-tensor
algorithm from a paper. It uses genuine third parameter derivatives of the
unchanged value-only objective, not third input derivatives and not merely
cubic regularization of a second-order Taylor model.

For parameter vector theta, gradient g and exact Hessian H:

1. Symmetrize H; choose a positive-definite damped matrix A = H + lambda I.
2. Compute a Newton direction d = -A^-1 g and apply a step-norm safeguard.
3. Hold d fixed and compute t = D^3 L(theta)[d,d,:] by nested autodiff.
4. Compute c = -0.5 A^-1 t. Limit its norm to half the Newton direction norm.
5. Backtrack both d+c and d on the actual training objective, retaining the
   best Armijo-acceptable decrease; adapt damping based on acceptance.

The third contraction requires no P-by-P-by-P tensor. The dense Hessian is
computed over parameter-coordinate chunks, not datapoint minibatches: all
10,000 training points contribute to every loss and derivative evaluation.
The matrix solve is an optimizer step, not a fitted linear output layer or
variable projection. There is no SciPy, symbolic fitting, or target-derivative
supervision in this pilot.

Related feasibility reference: Gower and Gower, *Higher-order Reverse Automatic
Differentiation with emphasis on the third-order*,
https://arxiv.org/abs/1309.5479 . The specific damping, correction clipping and
line-search safeguards here are our experimental implementation, not claimed
to reproduce their solver.

## Comparison protocol

- Three predeclared cases: 1/x on [0.5,2]; cos(x) on [-pi,pi];
  cos(x1)/x2 on [-pi,pi] x [0.5,2].
- Seeds 421–423, fresh 10k train, validation and test samples per seed.
- Input and target train-only standardization.
- Two private affine routes and three G=3 Gaussian maps including the output;
  sum for unary controls, product for the bivariate interaction.
- Frozen centers/widths with r(G)=0.2 G^(2/3). Parameters: 19 or 21.
- Common 10-outer-call seed-batched L-BFGS warm start; then branch into:
  one extra L-BFGS outer call (at most 20 inner steps), 20 damped Newton steps,
  and 20 safeguarded directional-third-order steps.
- L-BFGS continuation resets history at the branch; report this limitation.
- Higher-order branches process seeds individually; L-BFGS batches seeds.
  Thus timing compares these implementations, not optimized theoretical costs.
- Report both accuracy and measured wall time/oracle counts. Equal nominal
  step counts do not mean equal computation. Concurrent GPU work affects time.
- Verify Hessian symmetry and the third contraction against a central
  finite-difference Hessian-vector product before each case.
- All branches start with identical weights and frozen buffers. Test metrics
  are report-only; no test-driven optimizer or parameter selection.

The three-seed pilot is a bounded optimizer diagnostic, not a replacement for
the ten-seed production benchmark. Production PDE jobs and optimizers remain
unchanged. Checkpoints include the common starting parameters and frozen basis
buffers, final parameter vectors, RNG streams and normalization metadata.
