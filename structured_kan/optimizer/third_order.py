"""Small-model, safeguarded directional Chebyshev corrections.

This experimental optimizer uses genuine third derivatives of the scalar loss
with respect to parameters, not input derivatives or cubic-regularized Newton.
It does not form a P x P x P tensor. Damping and safeguards mean no claim of
global or cubic convergence is made for nonconvex neural objectives.
"""
import time
import torch


def third_contraction(loss, theta, direction):
    """Return D^3 loss(theta)[direction,direction,:], holding direction fixed."""
    direction = direction.detach()
    gradient = torch.func.grad(loss)
    def curvature(point):
        hv = torch.func.jvp(gradient, (point,), (direction,))[1]
        return (hv * direction).sum()
    return torch.func.grad(curvature)(theta).detach()


def minimize_small(loss, initial, *, steps=20, third_order=True, progress=None):
    theta = initial.detach().clone()
    gradient = torch.func.grad(loss)
    hessian = torch.func.jacrev(gradient, chunk_size=4)
    mu = 1e-6
    rows = []
    calls = dict(value=0, gradient=0, hessian=0, third_contraction=0)
    started = time.perf_counter()
    def value(point):
        calls['value'] += 1
        return float(loss(point).detach())
    current = value(theta)
    for iteration in range(steps):
        g = gradient(theta).detach(); calls['gradient'] += 1
        H = hessian(theta).detach(); calls['hessian'] += 1
        H = (H + H.T) / 2
        if not bool(torch.isfinite(H).all() & torch.isfinite(g).all()):
            raise RuntimeError('Nonfinite parameter derivatives')
        eig = torch.linalg.eigvalsh(H)
        scale = float(eig.abs().max().clamp_min(1e-12))
        damping = max(0., -float(eig[0])) + max(mu * scale, 1e-16)
        A = H + damping * torch.eye(theta.numel(), device=theta.device, dtype=theta.dtype)
        step = torch.linalg.solve(A, -g)
        radius = .25 * (1 + float(theta.norm()))
        step *= min(1., radius / max(float(step.norm()), 1e-30))
        directions = [('newton', step)]
        if third_order:
            contraction = third_contraction(loss, theta, step)
            calls['third_contraction'] += 1
            correction = torch.linalg.solve(A, -.5 * contraction)
            if bool(torch.isfinite(correction).all()):
                correction *= min(1., .5 * float(step.norm()) / max(float(correction.norm()), 1e-30))
                directions.insert(0, ('third', step + correction))
        best = current; best_theta = theta; accepted = 'none'; accepted_alpha = 0.
        for name, direction in directions:
            slope = float(g @ direction)
            if not bool(torch.isfinite(direction).all()) or slope >= 0:
                continue
            for backtrack in range(12):
                alpha = 2. ** (-backtrack)
                candidate = theta + alpha * direction
                trial = value(candidate)
                if trial <= current + 1e-4 * alpha * slope:
                    if trial < best:
                        best, best_theta, accepted, accepted_alpha = trial, candidate.detach(), name, alpha
                    break
        theta = best_theta
        current = best
        mu = max(1e-12, mu * .3) if accepted_alpha >= .5 else min(1e8, mu * 10.)
        row = dict(iteration=iteration+1, loss=current, accepted=accepted,
                   alpha=accepted_alpha, damping=damping, gradient_norm=float(g.norm()),
                   seconds=time.perf_counter()-started)
        rows.append(row)
        if progress:progress(row)
    torch.cuda.synchronize() if theta.is_cuda else None
    return theta, dict(seconds=time.perf_counter()-started, steps=steps, calls=calls,
                       accepted_third=sum(r['accepted']=='third' for r in rows), trace=rows)
