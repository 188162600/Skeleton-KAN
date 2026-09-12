"""Remote-only MLP checks, including seed batching versus individual gradients."""
import unittest
import torch
from ..model.MLP import MLP, architecture_sweep, parameter_count, parameter_hash
from ..dataset.regression import RegressionData, Split
from ..optimizer.fit import fit_seed_batch


class MLPTests(unittest.TestCase):
    def test_all_shapes_and_counts(self):
        configs = architecture_sweep()
        self.assertEqual(len(configs), 18)
        for d in (2, 3, 10):
            for arch in configs:
                model = MLP(d, arch['hidden_layers'], arch['hidden_width'], arch['activation'])
                self.assertEqual(sum(p.numel() for p in model.parameters()), parameter_count(d, arch['hidden_layers'], arch['hidden_width']))
                self.assertEqual(model(torch.randn(7, d, dtype=torch.float64)).shape, (7, 1))

    def test_seed_and_state(self):
        models = [MLP(3, 2, 32, 'tanh', seed=s) for s in range(421, 431)]
        self.assertEqual(len({parameter_hash(m) for m in models}), 10)
        same = MLP(3, 2, 32, 'tanh', seed=421)
        self.assertEqual(parameter_hash(same), parameter_hash(models[0]))
        for model in models:
            self.assertTrue(all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if n.endswith('bias')))
        x = torch.randn(5, 3, dtype=torch.float64)
        same.load_state_dict(models[-1].state_dict())
        self.assertTrue(torch.equal(same(x), models[-1](x)))

    def test_vmap_gradients(self):
        models = [MLP(2, 1, 4, 'silu', seed=s) for s in (421, 422)]
        params, buffers = torch.func.stack_module_state(models)
        x, y = torch.randn(2, 13, 2, dtype=torch.float64), torch.randn(2, 13, 1, dtype=torch.float64)
        def loss(p, b, a, target):
            return (torch.func.functional_call(models[0], (p, b), (a,))-target).square().mean()
        grads = torch.vmap(torch.func.grad(loss))(params, buffers, x, y)
        for i, model in enumerate(models):
            (model(x[i])-y[i]).square().mean().backward()
            for name, p in model.named_parameters():
                torch.testing.assert_close(grads[name][i], p.grad, rtol=1e-12, atol=1e-12)

    def test_seed_batched_lbfgs(self):
        x = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(-1, 1)
        split = Split(x, 0.4*x+0.2)
        data = RegressionData(split, split, split, 'test', 421)
        models = [MLP(1, 1, 4, 'tanh', seed=s) for s in (421, 422)]
        before = [float((m(x)-split.y).square().mean().detach()) for m in models]
        result = fit_seed_batch(models, [data, data], outer_steps=2)
        self.assertEqual(len(result.inner_iterations), 2)
        for model, old in zip(models, before):
            self.assertLess(float((model(x)-split.y).square().mean().detach()), old)


if __name__ == '__main__':
    unittest.main()
