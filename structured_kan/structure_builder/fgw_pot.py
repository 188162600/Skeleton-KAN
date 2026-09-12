"""Modern POT FGW numerical backend; no native graph/phi clustering shortcuts.

Raw squared feature distances enter POT, which applies (1-alpha) itself.
Keep the published source-initialized outer loop in fgw_source_initialized.
The POT barycenter update order and convergence implementation are deliberately
not represented as bitwise-equivalent to the older authors' repository.
"""
from __future__ import annotations
import hashlib
import inspect
from pathlib import Path
import numpy as np
import ot
from scipy.spatial.distance import cdist


def metadata():
    source = Path(inspect.getfile(ot.gromov.fgw_barycenters))
    return dict(backend='POT unregularized conditional-gradient FGW',
                pot_version=ot.__version__, pot_solver_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                armijo=False, warmstartT=True, entropic_regularization=False,
                assignment_max_iter=500, assignment_tol_abs=1e-9, assignment_tol_rel=0.0,
                barycenter_max_iter=100, barycenter_inner_max_iter=100,
                barycenter_tol=1e-9, barycenter_inner_tol_rel=1e-5,
                barycenter_stopping='POT barycenter: either feature or structure change <= tol; maximum iterations unchanged',
                update_order='POT: transport then feature/structure; old implementation used feature/structure then transport',
                feature_cost='raw squared Euclidean; (1-alpha) applied exactly once inside POT',
                execution='float64 CPU; numerical-library thread limits supplied by launcher')


def transport(features_a, structure_a, features_b, structure_b, alpha=.5, G0=None):
    # Do not multiply this matrix by (1-alpha), unlike the old fgw_lp API.
    costs = cdist(features_a, features_b, metric='sqeuclidean')
    p = np.full(len(structure_a), 1.0/len(structure_a))
    q = np.full(len(structure_b), 1.0/len(structure_b))
    plan, log = ot.gromov.fused_gromov_wasserstein(
        costs, structure_a, structure_b, p, q, loss_fun='square_loss',
        symmetric=True, alpha=alpha, armijo=False, G0=G0, log=True,
        max_iter=500, tol_rel=0.0, tol_abs=1e-9)
    value = float(log['fgw_dist'])
    if not np.isfinite(value) or not np.isfinite(plan).all():
        raise FloatingPointError('Nonfinite POT FGW solve')
    error = max(float(np.abs(plan.sum(1)-p).max()), float(np.abs(plan.sum(0)-q).max()))
    if error >= 1e-7 or plan.min() < -1e-10:
        raise ValueError(f'Infeasible POT transport: marginal error {error}')
    return plan, log


def cdist_fgw(X_features, X_structure, Y_features, Y_structure, alpha, metric='sqeuclidean'):
    if metric != 'sqeuclidean':
        raise ValueError('The frozen categorical FGW protocol uses squared Euclidean features')
    distances = np.empty((len(X_features), len(Y_features)))
    for i, (x, cx) in enumerate(zip(X_features, X_structure)):
        for j, (y, cy) in enumerate(zip(Y_features, Y_structure)):
            _, log = transport(x, cx, y, cy, alpha)
            distances[i,j] = float(log['fgw_dist'])
    return distances


def fgw_barycenters(*, N, Ys, Cs, ps, lambdas, alpha, max_iter=100,
                    fixed_structure=False, fixed_features=False,
                    init_X=None, init_C=None, verbose=False):
    return ot.gromov.fgw_barycenters(
        N=N, Ys=Ys, Cs=Cs, ps=ps, lambdas=lambdas, alpha=alpha,
        max_iter=max_iter, tol=1e-9, stop_criterion='barycenter',
        armijo=False, symmetric=True, warmstartT=True, log=True,
        fixed_structure=fixed_structure, fixed_features=fixed_features,
        init_X=init_X, init_C=init_C, verbose=verbose, random_state=421)


def native_fgw_pot(trees, k, vendor, out):
    import sys
    from .fgw import native_fgw_source
    return native_fgw_source(trees, k, vendor, out, numerical=sys.modules[__name__])
