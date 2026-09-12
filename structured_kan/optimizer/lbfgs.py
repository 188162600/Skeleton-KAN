"""Project standalone, single-model L-BFGS with optimizer.step(closure).

Preserves the earlier project optimizer, including nosync_fma and coalesced
strong-Wolfe options. This is not a wrapper around torch.optim.LBFGS or the
independent-seed controller. Its PyKAN/PyTorch-derived implementation retains
the upstream attribution documented in THIRD_PARTY_NOTICES.md.
"""

import math
import torch
from functools import reduce
from torch.optim import Optimizer

__all__ = ['LBFGS']


def _coalesced_host_scalars(*values):
    """Copy several scalar tensors to the host with one synchronization."""
    if not values:
        return ()
    packed = torch.stack(tuple(value.detach().reshape(()) for value in values))
    return tuple(packed.tolist())

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    if x1 == x2:
        return (xmin_bound + xmax_bound) / 2.
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    # The original tensor-valued path is retained for compatibility.  The
    # coalesced/no-sync line search passes Python floats and needs the scaled
    # calculation below because Python raises on overflow instead of yielding
    # ``inf`` as Torch does.
    if any(torch.is_tensor(value) for value in (d1, g1, g2)):
        d2_square = d1**2 - g1 * g2
        if d2_square >= 0:
            d2 = d2_square.sqrt()
            if x1 <= x2:
                min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
            else:
                min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
            return min(max(min_pos, xmin_bound), xmax_bound)
        return (xmin_bound + xmax_bound) / 2.

    scalar_values = (float(d1), float(g1), float(g2))
    if not all(math.isfinite(value) for value in scalar_values):
        return (xmin_bound + xmax_bound) / 2.
    # Normalize before squaring.  This is algebraically identical to
    # d1**2-g1*g2, but avoids Python-float overflow on a rejected trial whose
    # derivative is extremely large.
    scale = max(abs(scalar_values[0]), abs(scalar_values[1]), abs(scalar_values[2]), 1.0)
    d1_normalized = scalar_values[0] / scale
    g1_normalized = scalar_values[1] / scale
    g2_normalized = scalar_values[2] / scale
    d2_square = d1_normalized**2 - g1_normalized * g2_normalized
    if d2_square >= 0:
        d2_normalized = math.sqrt(d2_square)
        if x1 <= x2:
            denominator = g2_normalized - g1_normalized + 2 * d2_normalized
            if abs(denominator) < 1.0e-300:
                return (xmin_bound + xmax_bound) / 2.
            ratio = (g2_normalized + d2_normalized - d1_normalized) / denominator
            min_pos = x2 - (x2 - x1) * ratio
        else:
            denominator = g1_normalized - g2_normalized + 2 * d2_normalized
            if abs(denominator) < 1.0e-300:
                return (xmin_bound + xmax_bound) / 2.
            ratio = (g1_normalized + d2_normalized - d1_normalized) / denominator
            min_pos = x1 - (x1 - x2) * ratio
        if not math.isfinite(float(min_pos)):
            return (xmin_bound + xmax_bound) / 2.
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old low becomes new high
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals


def _strong_wolfe_coalesced(obj_func,
                             x,
                             t,
                             d,
                             f,
                             g,
                             gtd,
                             g_max,
                             d_norm,
                             c1=1e-4,
                             c2=0.9,
                             tolerance_change=1e-9,
                             max_ls=25):
    """Strong Wolfe with all line-search control scalars held on the host.

    ``obj_func`` coalesces the loss, directional derivative, and gradient
    maximum into one device-to-host transfer per closure. The line-search
    equations and branching are otherwise the same as ``_strong_wolfe``.
    """
    g = g.clone(memory_format=torch.contiguous_format)
    f_new, g_new, gtd_new, g_max_new = obj_func(x, t, d)
    ls_func_evals = 1

    t_prev, f_prev, g_prev, gtd_prev, g_max_prev = 0.0, f, g, gtd, g_max
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            bracket_g_max = [g_max_prev, g_max_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            bracket_gtd = [gtd_new]
            bracket_g_max = [g_max_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            bracket_g_max = [g_max_prev, g_max_new]
            break

        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        g_max_prev = g_max_new
        f_new, g_new, gtd_new, g_max_new = obj_func(x, t, d)
        ls_func_evals += 1
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]
        bracket_g_max = [g_max, g_max_new]

    insuf_progress = False
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new, gtd_new, g_max_new = obj_func(x, t, d)
        ls_func_evals += 1
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            bracket_g_max[high_pos] = g_max_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
                bracket_g_max[high_pos] = bracket_g_max[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new
            bracket_g_max[low_pos] = g_max_new

    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    g_max_new = bracket_g_max[low_pos]
    return f_new, g_new, t, ls_func_evals, g_max_new



class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
        two_loop_mode (str): ``reference`` preserves the original scalar-alpha
            updates; ``nosync_fma`` keeps the two-loop scalar tensors on device
            using fused addcmul updates; ``compact_solve`` applies the same
            history through triangular solves (default: ``reference``).
        scalar_mode (str): ``reference`` preserves CUDA scalar control;
            ``coalesced_host`` performs one packed scalar read per dependency
            point and runs strong-Wolfe scalar algebra on the host.
    """

    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn=None,
                 two_loop_mode="reference",
                 scalar_mode="reference"):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        if two_loop_mode not in ("reference", "nosync_fma", "compact_solve"):
            raise ValueError(
                "two_loop_mode must be 'reference', 'nosync_fma', or "
                "'compact_solve'")
        if scalar_mode not in ("reference", "coalesced_host"):
            raise ValueError("scalar_mode must be 'reference' or 'coalesced_host'")
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            tolerance_ys=tolerance_ys,
            history_size=history_size,
            line_search_fn=line_search_fn,
            two_loop_mode=two_loop_mode,
            scalar_mode=scalar_mode)
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = float(closure())
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite task loss")
        return loss, flat_grad

    def _directional_evaluate_coalesced(self, closure, x, t, d):
        self._add_grad(t, d)
        loss = closure()
        flat_grad = self._gather_flat_grad()
        directional_derivative = flat_grad.dot(d)
        gradient_max = flat_grad.abs().max()
        packed = _coalesced_host_scalars(
            loss, directional_derivative, gradient_max)
        self._set_param(x)
        loss_value, directional_value, gradient_max_value = packed
        if not math.isfinite(loss_value):
            raise FloatingPointError("non-finite task loss")
        return loss_value, flat_grad, directional_value, gradient_max_value


    @torch.no_grad()
    def step(self, closure):
        """Perform a single optimization step.

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        tolerance_ys = group['tolerance_ys']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']
        two_loop_mode = group['two_loop_mode']
        scalar_mode = group['scalar_mode']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)
        debug_stop_history = state.get('_debug_stop_history')
        stop_reason = None

        # evaluate initial f(x) and df/dx
        orig_loss = closure()
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        if scalar_mode == "coalesced_host":
            loss, gradient_max, gradient_l1 = _coalesced_host_scalars(
                orig_loss, flat_grad.abs().max(), flat_grad.abs().sum())
            opt_cond = gradient_max <= tolerance_grad
        else:
            loss = float(orig_loss)
            gradient_max = flat_grad.abs().max()
            gradient_l1 = None
            opt_cond = gradient_max <= tolerance_grad
        if not math.isfinite(loss):
            raise FloatingPointError("non-finite task loss")

        # optimal condition
        if opt_cond:
            if debug_stop_history is not None:
                debug_stop_history.append({'reason': 'initial_optimality', 'inner_iterations': 0,
                                           'current_evals': current_evals, 'loss': loss,
                                           'gradient_max': (
                                               gradient_max if scalar_mode == "coalesced_host"
                                               else gradient_max.item())})
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                ys_control = (
                    _coalesced_host_scalars(ys)[0]
                    if scalar_mode == "coalesced_host" else ys
                )
                if ys_control > tolerance_ys:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if two_loop_mode == "compact_solve":
                    # The two recurrences can be written as one upper and one
                    # lower triangular solve. S^T Y contains every history
                    # cross-product required by both recurrences.
                    directions = torch.stack(old_dirs, dim=1)
                    steps = torch.stack(old_stps, dim=1)
                    rho = torch.stack(ro)
                    sty = steps.transpose(0, 1).matmul(directions)
                    identity = torch.eye(
                        num_old, dtype=flat_grad.dtype, device=flat_grad.device)

                    q0 = flat_grad.neg()
                    upper = identity + torch.triu(sty, diagonal=1) * rho[:, None]
                    alpha = torch.linalg.solve_triangular(
                        upper,
                        (rho * steps.transpose(0, 1).matmul(q0)).unsqueeze(1),
                        upper=True,
                    ).squeeze(1)
                    q = q0 - directions.matmul(alpha)

                    initial = q * H_diag
                    lower = (
                        identity
                        + torch.tril(sty.transpose(0, 1), diagonal=-1)
                        * rho[:, None]
                    )
                    correction = torch.linalg.solve_triangular(
                        lower,
                        (alpha - rho * directions.transpose(0, 1).matmul(initial))
                        .unsqueeze(1),
                        upper=False,
                    ).squeeze(1)
                    d = initial + steps.matmul(correction)
                else:
                    if 'al' not in state:
                        state['al'] = [None] * history_size
                    al = state['al']

                    # iteration in L-BFGS loop collapsed to use just one buffer
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_stps[i].dot(q) * ro[i]
                        if two_loop_mode == "nosync_fma":
                            # ``alpha=`` converts a zero-dimensional CUDA tensor
                            # to a host scalar, synchronising once per history item.
                            # addcmul keeps it on device and is intentionally a
                            # fused operation, so it is numerically near--not
                            # bitwise--identical to the reference path.
                            q.addcmul_(old_dirs[i], al[i], value=-1.0)
                        else:
                            q.add_(old_dirs[i], alpha=-al[i])

                    # multiply by initial Hessian
                    # r/d is the final direction
                    d = r = torch.mul(q, H_diag)
                    for i in range(num_old):
                        be_i = old_dirs[i].dot(r) * ro[i]
                        if two_loop_mode == "nosync_fma":
                            r.addcmul_(old_stps[i], al[i] - be_i, value=1.0)
                        else:
                            r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                if scalar_mode == "coalesced_host":
                    t = min(1., 1. / gradient_l1) * lr
                else:
                    t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d
            direction_norm = None
            if scalar_mode == "coalesced_host":
                gtd, direction_norm = _coalesced_host_scalars(gtd, d.abs().max())

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                stop_reason = 'directional_derivative'
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    if scalar_mode == "coalesced_host":
                        def obj_func(x, t, d):
                            return self._directional_evaluate_coalesced(
                                closure, x, t, d)

                        loss, flat_grad, t, ls_func_evals, gradient_max = (
                            _strong_wolfe_coalesced(
                            obj_func, x_init, t, d, loss, flat_grad, gtd,
                            gradient_max, direction_norm,
                            tolerance_change=tolerance_change)
                        )
                    else:
                        def obj_func(x, t, d):
                            return self._directional_evaluate(closure, x, t, d)

                        loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                            obj_func, x_init, t, d, loss, flat_grad, gtd,
                            tolerance_change=tolerance_change)
                self._add_grad(t, d)
                if scalar_mode == "coalesced_host":
                    opt_cond = gradient_max <= tolerance_grad
                else:
                    gradient_max = flat_grad.abs().max()
                    opt_cond = gradient_max <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    with torch.enable_grad():
                        new_loss = closure()
                    flat_grad = self._gather_flat_grad()
                    if scalar_mode == "coalesced_host":
                        loss, gradient_max = _coalesced_host_scalars(
                            new_loss, flat_grad.abs().max())
                        opt_cond = gradient_max <= tolerance_grad
                    else:
                        loss = float(new_loss)
                        gradient_max = flat_grad.abs().max()
                        opt_cond = gradient_max <= tolerance_grad
                    if not math.isfinite(loss):
                        raise FloatingPointError("non-finite task loss")
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                stop_reason = 'max_iter'
                break

            if current_evals >= max_eval:
                stop_reason = 'max_eval'
                break

            # optimal condition
            if opt_cond:
                stop_reason = 'optimality'
                break

            # lack of progress
            step_max = (
                abs(t) * direction_norm
                if scalar_mode == "coalesced_host"
                else d.mul(t).abs().max()
            )
            if step_max <= tolerance_change:
                stop_reason = 'parameter_change'
                break

            if abs(loss - prev_loss) < tolerance_change:
                stop_reason = 'loss_change'
                break

        if debug_stop_history is not None:
            if scalar_mode == "coalesced_host":
                debug_gradient_max = gradient_max
                debug_step_max = abs(t) * direction_norm
                debug_gtd = gtd
            else:
                debug_gradient_max = flat_grad.abs().max().item()
                debug_step_max = d.mul(t).abs().max().item()
                debug_gtd = gtd.item()
            debug_stop_history.append({'reason': stop_reason or 'unknown', 'inner_iterations': n_iter,
                                       'current_evals': current_evals, 'loss': loss,
                                       'gradient_max': debug_gradient_max,
                                       'step_max': debug_step_max,
                                       'directional_derivative': debug_gtd})

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss





