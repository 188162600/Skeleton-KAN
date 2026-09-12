"""Single-model closure, optimizer state and resume regression checks."""
import copy
import unittest

import torch

from structured_kan.optimizer.lbfgs import LBFGS


class StandaloneLBFGSTest(unittest.TestCase):
    def problem(self, mode='nosync_fma', scalar_mode='coalesced_host', max_iter=20):
        x = torch.nn.Parameter(torch.tensor([3., -4., 2.], dtype=torch.float64))
        optimizer = LBFGS([x], max_iter=max_iter, history_size=10,
            tolerance_grad=1e-15, tolerance_change=1e-20,
            line_search_fn='strong_wolfe', two_loop_mode=mode, scalar_mode=scalar_mode)
        target = x.detach().new_tensor([0.5, 1., -2.])
        weights = x.detach().new_tensor([1., 2., 4.])

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = ((x-target).square()*weights).sum()
            loss.backward()
            return loss
        return x, optimizer, closure, target

    def test_closure_modes(self):
        for mode, scalars in [('reference', 'reference'), ('nosync_fma', 'coalesced_host')]:
            with self.subTest(mode=mode):
                x, opt, closure, target = self.problem(mode, scalars)
                for _ in range(3):
                    returned = opt.step(closure)
                    self.assertTrue(torch.is_tensor(returned))
                torch.testing.assert_close(x, target, rtol=0., atol=1e-9)
                self.assertGreater(opt.state[x]['func_evals'], 0)

    def test_state_dict_resume(self):
        x, opt, closure, _ = self.problem(max_iter=2)
        opt.step(closure)
        saved_x, saved_optimizer = x.detach().clone(), copy.deepcopy(opt.state_dict())
        opt.step(closure)
        expected = x.detach().clone()
        resumed, resumed_opt, resumed_closure, _ = self.problem(max_iter=2)
        with torch.no_grad():
            resumed.copy_(saved_x)
        resumed_opt.load_state_dict(saved_optimizer)
        resumed_opt.step(resumed_closure)
        torch.testing.assert_close(resumed, expected, rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main()
