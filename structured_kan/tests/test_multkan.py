"""Remote-only checks against native PyKAN; no target-conditioned construction."""
from collections import Counter
import json
import unittest

import torch

from ..model.MultKAN import MultKAN, architecture_sweep, native_model, parameter_count, widths
from ..model.MLP import parameter_hash
from ..model.neural_baselines import build_model
from ..scripts.neural_benchmark import inspect_config
from ..dataset.regression import RegressionData, Split
from ..optimizer.fit import fit_seed_batch


def args(arch):
    return {k: arch[k] for k in ('first_hidden', 'second_hidden', 'grid')}


class MultKANTests(unittest.TestCase):
    def test_actual_parameter_audit(self):
        _, config, rows = inspect_config('multkan18_domain80.json')
        histogram = Counter(r['variables'] for r in rows)
        by_dimension = {}
        for d in sorted(histogram):
            counts = []
            for arch in architecture_sweep():
                model = build_model('multkan', d, arch)
                value = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.assertEqual(value, parameter_count(d, **args(arch)))
                trainable = {n for n, p in model.named_parameters() if p.requires_grad}
                self.assertTrue(all(n.endswith(('coef', 'scale_base', 'scale_sp')) for n in trainable))
                self.assertEqual(len(trainable), 3*(len(widths(d, arch['first_hidden'], arch['second_hidden']))-1))
                counts.append(value)
            by_dimension[d] = counts
        mean = sum(sum(by_dimension[d])*n for d,n in histogram.items())/(80*18)
        print(json.dumps(dict(multkan_parameter_audit=dict(
            dimension_histogram=dict(histogram), counts_by_dimension=by_dimension,
            mean_trainable_parameters=mean, equations=80, configs=18))), flush=True)

    def test_native_forward_gradient_and_state(self):
        for d in (2, 3, 9):
            for arch in architecture_sweep():
                native = native_model(d, **args(arch), optimized=False)
                fast = build_model('multkan', d, arch)
                # Compare at exactly the same native state, not two least-squares
                # random-spline initializations with platform-dependent roundoff.
                keys = fast.network.load_state_dict(native.state_dict(), strict=False)
                self.assertFalse(keys.missing_keys)
                self.assertFalse(keys.unexpected_keys)
                for name, value in native.state_dict().items():
                    torch.testing.assert_close(value, fast.network.state_dict()[name], rtol=0, atol=0)
                x = torch.linspace(-4, 4, 31*d, dtype=torch.float64).reshape(31,d)
                # Include knot locations and the boundary of the extended grid.
                knot_x = native.act_fun[0].grid.T.contiguous()
                x = torch.cat([x, knot_x])
                xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
                ya, yb = native(xa), fast(xb)
                torch.testing.assert_close(ya, yb, atol=1e-11, rtol=1e-11)
                ya.square().mean().backward(); yb.square().mean().backward()
                torch.testing.assert_close(xa.grad, xb.grad, atol=1e-10, rtol=1e-10)
                fast_params = dict(fast.network.named_parameters())
                for name,p in native.named_parameters():
                    if p.requires_grad:
                        actual = fast_params[name].grad
                        if name.endswith('.coef'):
                            actual = actual.permute(0,2,1)
                        torch.testing.assert_close(p.grad, actual, atol=1e-10, rtol=1e-10)
                replay = build_model('multkan', d, arch, seed=430)
                replay.load_state_dict(fast.state_dict())
                torch.testing.assert_close(replay(x), fast(x), atol=0, rtol=0)

    def test_seeded_initializations(self):
        arch = architecture_sweep()[4]
        models = [build_model('multkan',3,arch,seed=s) for s in range(421,431)]
        self.assertEqual(len({parameter_hash(m) for m in models}),10)
        same=build_model('multkan',3,arch,seed=421)
        for name,value in same.state_dict().items():
            torch.testing.assert_close(value,models[0].state_dict()[name],atol=1e-12,rtol=1e-12)

    def test_vmap_gradients(self):
        for index in (0,4,8,17):
            models=[build_model('multkan',3,architecture_sweep()[index],seed=s) for s in (421,422)]
            params,buffers=torch.func.stack_module_state(models)
            x=torch.randn(2,13,3,dtype=torch.float64);y=torch.randn(2,13,1,dtype=torch.float64)
            def loss(p,b,x,y):
                return (torch.func.functional_call(models[0],(p,b),(x,))-y).square().mean()
            grads=torch.vmap(torch.func.grad(loss))(params,buffers,x,y)
            for i,model in enumerate(models):
                (model(x[i])-y[i]).square().mean().backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(grads[name][i],p.grad,atol=1e-11,rtol=1e-11)

    def test_seed_batched_lbfgs(self):
        x=torch.linspace(-1,1,24,dtype=torch.float64).reshape(-1,1)
        split=Split(x,0.4*x+0.2)
        data=RegressionData(split,split,split,'test',421)
        arch=architecture_sweep()[3]
        models=[build_model('multkan',1,arch,seed=s) for s in (421,422)]
        before=[float((m(x)-split.y).square().mean().detach()) for m in models]
        result=fit_seed_batch(models,[data,data],outer_steps=2)
        self.assertEqual(len(result.inner_iterations),2)
        for model,old in zip(models,before):
            self.assertLess(float((model(x)-split.y).square().mean().detach()),old)


if __name__=='__main__':unittest.main()
