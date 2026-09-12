"""Strong-Wolfe control and interpolation; relocated without formula changes."""
import math
import torch

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
