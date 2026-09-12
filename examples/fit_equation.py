"""Fit the benchmark Cesaro equation exp(x0/x1)/x1 with its selected skeleton.

Run from the repository root: python -m examples.fit_equation --device cuda
This is an API example, not an 18-candidate, ten-seed benchmark.
"""
import torch

from structured_kan.dataset.fresh_sampling import draw_unit
from .common import (ROOT, arguments, benchmark_equation, build_model, check_output, fit, make_optimizer,
                     metrics, save, standardization)


def main():
    args = arguments(__doc__, 'example_equation', default_builder='IAPB05', default_points=10000)
    check_output(args)
    row, reference = benchmark_equation()
    if args.builder != reference['builder']:
        raise ValueError('this example uses the benchmark-selected IAPB05 and its archived per-map budgets')
    bounds = torch.tensor(row['quality_audit']['support']['bounds'], dtype=torch.float64)

    def sample(split):
        unit = draw_unit(row['case_id'], args.seed, split, args.points, 2)
        x = bounds[:, 0] + (bounds[:, 1]-bounds[:, 0])*torch.from_numpy(unit)
        x = x.to(args.device)
        y = torch.exp(x[:, :1]/x[:, 1:2])/x[:, 1:2]
        return x, y

    x_train, y_train = sample('train')
    x_val, y_val = sample('validation')
    x_test, y_test = sample('test')
    x_mean, x_std = standardization(x_train)
    y_mean, y_std = standardization(y_train)
    x_fit, y_fit = (x_train-x_mean)/x_std, (y_train-y_mean)/y_std
    model, spec, metadata = build_model(args, x_fit, bank_path=ROOT/reference['catalogue'],
                                      route_budgets=reference['private_phi_G_by_route'])
    assert metadata['parameters'] == reference['parameters']
    optimizer = make_optimizer(model)

    def predict(x):
        return model((x-x_mean)/x_std)*y_std + y_mean

    # An ordinary single-model closure: zero_grad, forward, backward, loss.
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x_fit)-y_fit).square().mean()
        loss.backward()
        return loss

    @torch.no_grad()
    def validation_loss():
        return metrics(predict(x_val), y_val)['nmse']

    best_step = fit(model, optimizer, closure, validation_loss, args.outer_steps)
    with torch.no_grad():
        result = dict(task='equation', case_id=row['case_id'], equation=row['expression'],
                      support=bounds.tolist(), benchmark_reference=reference['reference'],
                      points_per_split=args.points, selected_outer_step=best_step,
                      train=metrics(predict(x_train), y_train),
                      validation=metrics(predict(x_val), y_val),
                      test=metrics(predict(x_test), y_test))
    save(args, model, spec, metadata,
         dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std), result)


if __name__ == '__main__':
    main()
