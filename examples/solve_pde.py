"""Solve u_t - 0.1*u_xx = 0 on (x,t) in [0,1]^2 using a PINN loss.

u(x,0)=sin(pi*x); u(0,t)=u(1,t)=0.
Exact interior values are used for validation/test only, never training.
Run from the repository root: python -m examples.solve_pde --device cuda
"""
import math

import torch

from .common import (arguments, build_model, check_output, fit, make_optimizer,
                     metrics, save, standardization)


def main():
    args = arguments(__doc__, 'example_heat')
    check_output(args)
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    nu = 0.1

    def sample(n):
        return torch.rand(n, 2, dtype=torch.float64, generator=generator).to(args.device)

    def exact(xt):
        return torch.exp(-nu*math.pi**2*xt[:, 1:2])*torch.sin(math.pi*xt[:, :1])

    interior = sample(args.points)
    n_constraint = max(4, args.points//4)
    initial, left, right = (sample(n_constraint) for _ in range(3))
    initial[:, 1] = 0.
    left[:, 0], right[:, 0] = 0., 1.
    constraints = torch.cat([initial, left, right])
    values = torch.cat([torch.sin(math.pi*initial[:, :1]),
                        left.new_zeros(n_constraint, 1), right.new_zeros(n_constraint, 1)])
    validation, test = sample(args.points), sample(args.points)
    x_mean, x_std = standardization(interior)
    # Only prescribed IC/BC values determine output normalization.
    y_mean, y_std = standardization(values)
    model, spec, metadata = build_model(args, (interior-x_mean)/x_std)
    optimizer = make_optimizer(model)

    def predict(xt):
        return model((xt-x_mean)/x_std)*y_std + y_mean

    def residual(xt):
        # Differentiation includes normalization, so derivatives are physical.
        xt = xt.detach().requires_grad_(True)
        u = predict(xt)
        grad = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_t, u_x = grad[:, 1:2], grad[:, :1]
        u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, :1]
        return u_t - nu*u_xx

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (0.01*residual(interior).square().mean()
                + (predict(constraints)-values).square().mean()) / y_std.square().squeeze()
        loss.backward()
        return loss

    @torch.no_grad()
    def validation_loss():
        return metrics(predict(validation), exact(validation))['nmse']

    best_step = fit(model, optimizer, closure, validation_loss, args.outer_steps)
    residual_mse = float(residual(test).square().mean().detach())
    with torch.no_grad():
        result = dict(task='heat_pde', pde='u_t - 0.1*u_xx = 0',
            exact_solution='exp(-0.1*pi**2*t)*sin(pi*x)',
            support=[[0., 1.], [0., 1.]], interior_points=args.points,
            points_per_constraint=n_constraint, selected_outer_step=best_step,
            residual_weight=0.01, interior_training_labels=False,
            test_residual_mse=residual_mse,
            validation=metrics(predict(validation), exact(validation)),
            test=metrics(predict(test), exact(test)))
    save(args, model, spec, metadata,
         dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std), result)


if __name__ == '__main__':
    main()
