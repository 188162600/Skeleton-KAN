"""Small deterministic forward/gradient tests; no fitting or dataset downloads."""
import copy
import unittest

import torch
from torch import nn

from structured_kan import GaussianBasis, KAN, StructuredKAN, StructuredKANBuilder
from structured_kan.model.GaussianBasis import default_width_ratio


class Square(nn.Module):
    def forward(self, x):
        return x.square()


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        self.builder = StructuredKANBuilder(2)
        self.x = torch.linspace(-1, 1, 40).reshape(20, 2)

    def inputs(self):
        return [self.builder.input(0), self.builder.input(1)]

    def test_phi_is_applied_before_sum(self):
        model = StructuredKAN(self.inputs(), [Square(), Square()], "sum")
        expected = self.x[:, :1].square() + self.x[:, 1:].square()
        self.assertTrue(torch.equal(model(self.x), expected))
        self.assertFalse(torch.equal(model(self.x), self.x.sum(-1, keepdim=True).square()))

    def test_phi_is_applied_before_product(self):
        model = StructuredKAN(self.inputs(), [Square(), nn.Identity()], "prod")
        self.assertTrue(torch.equal(model(self.x), self.x[:, :1].square()*self.x[:, 1:]))

    def test_nested_parent_owns_its_phi(self):
        inner = StructuredKAN(self.inputs(), [nn.Identity(), Square()], "prod")
        model = StructuredKAN([self.builder.input(0), inner], [nn.Identity(), Square()], "sum")
        expected = self.x[:, :1] + (self.x[:, :1]*self.x[:, 1:].square()).square()
        self.assertTrue(torch.equal(model(self.x), expected))

    def test_unary_sum_and_product_identical(self):
        a = StructuredKAN([self.builder.input(0)], [Square()], "sum")
        b = StructuredKAN([self.builder.input(0)], [Square()], "prod")
        self.assertTrue(torch.equal(a(self.x), b(self.x)))

    def test_no_implicit_output_or_node_parameters(self):
        node = StructuredKAN(self.inputs(), [nn.Identity(), nn.Identity()])
        self.assertEqual(node.parameter_count(), 0)
        self.assertEqual(list(node.parameters()), [])

    def test_invalid_arity(self):
        for children, maps in (([], []), (self.inputs(), []), (self.inputs(), [Square()])):
            with self.assertRaises(ValueError):
                StructuredKAN(children, maps)
        with self.assertRaises(ValueError):
            StructuredKAN(self.inputs(), [Square(), Square()], "subtract")
        with self.assertRaises(TypeError):
            StructuredKAN({"input": 0}, [Square()])

    def test_pytorch_child_traversal_not_shadowed(self):
        node = StructuredKAN(self.inputs(), [self.builder.unary(3), self.builder.unary(5)])
        self.assertTrue(callable(node.children))
        node.float().eval()
        self.assertEqual(node(self.x.float()).dtype, torch.float32)
        self.assertFalse(node.phi[0].training)

    def test_basis_formula_and_frozen_defaults(self):
        basis = GaussianBasis([-1., 0., 1.], .5)
        expected = torch.exp(-.5*((self.x.unsqueeze(-1)-basis.centers)/.5).square())
        self.assertTrue(torch.equal(basis(self.x), expected))
        self.assertEqual(list(basis.parameters()), [])
        self.assertEqual(set(dict(basis.named_buffers())), {"centers", "width"})

    def test_G_counts_and_width_rule(self):
        for g in (3, 5, 9, 15, 30):
            basis = GaussianBasis.uniform(g)
            edge = KAN(basis)
            self.assertEqual(sum(p.numel() for p in edge.parameters()), g+2)
            self.assertEqual(edge.G, g)
            self.assertAlmostEqual(float(basis.width), default_width_ratio(g)*2/(g-1))
            self.assertTrue(torch.equal(edge(self.x), self.x))

    def test_learnable_grid_requires_opt_in(self):
        edge = KAN(GaussianBasis.uniform(5, learn_centers=True, learn_width=True))
        self.assertEqual(sum(p.numel() for p in edge.parameters()), 2*5+3)

    def test_invalid_grid(self):
        for g in (1, 0, True, 3.5):
            with self.assertRaises(ValueError):
                GaussianBasis.uniform(g)
        for width in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                GaussianBasis([0, 1], width)

    def test_KAN_affine_plus_gaussian(self):
        basis = GaussianBasis.uniform(5)
        edge = KAN(basis, coefficients=[.1, .2, -.1, .4, -.3], a=2., b=3.)
        expected = 2*self.x+3+torch.einsum('...g,g->...', basis(self.x), edge.coefficients)
        self.assertTrue(torch.equal(edge(self.x), expected))

    def test_shared_phi_counts_once_and_load_preserves_alias(self):
        phi = self.builder.unary(3)
        model = StructuredKAN(self.inputs(), [phi, phi])
        self.assertEqual(model.parameter_count(), 5)
        other = copy.deepcopy(model)
        other.load_state_dict(model.state_dict())
        self.assertIs(other.phi[0], other.phi[1])
        self.assertTrue(torch.equal(other(self.x), model(self.x)))

    def test_distinct_phi_same_grid_not_tied(self):
        grid = GaussianBasis.uniform(3)
        model = StructuredKAN(self.inputs(), [KAN(grid), KAN(grid)])
        self.assertEqual(model.parameter_count(), 10)
        self.assertIsNot(model.phi[0].coefficients, model.phi[1].coefficients)

    def test_seed_batch_vmap_and_gradients(self):
        phi = self.builder.unary(3)
        model = StructuredKAN(self.inputs(), [phi, phi], "prod")
        with torch.no_grad():
            phi.coefficients.copy_(torch.tensor([.1, -.2, .3]))
        models = [copy.deepcopy(model) for _ in range(3)]
        params, buffers = torch.func.stack_module_state(models)
        values = torch.vmap(lambda p, b: torch.func.functional_call(models[0], (p,b), (self.x,), tie_weights=False))(params, buffers)
        torch.testing.assert_close(values, torch.stack([m(self.x) for m in models]), rtol=0, atol=0)
        grads = torch.vmap(torch.func.grad(lambda p,b:
            torch.func.functional_call(models[0], (p,b), (self.x,), tie_weights=False).square().mean()))(params,buffers)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(all(torch.isfinite(m(self.x)).all() for m in models))

    def test_builder_dynamic_G(self):
        spec = dict(operation='sum', children=[{'input':0}, {'input':1}], phi=[3, 9])
        model = self.builder.build(spec)
        self.assertEqual([edge.G for edge in model.phi], [3, 9])
        self.assertEqual(model.parameter_count(), 16)
        self.assertTrue(torch.equal(model(self.x), self.x.sum(-1, keepdim=True)))

    def test_builder_shared_routes_and_phi(self):
        route = {'affine':[1., 2.], 'share':'r'}
        edge = {'G':5, 'share':'p'}
        model = self.builder.build(dict(operation='sum', children=[route,route], phi=[edge,edge]))
        self.assertIs(model.branches[0], model.branches[1])
        self.assertIs(model.phi[0], model.phi[1])
        self.assertEqual(model.parameter_count(), 3+7)

    def test_separate_builds_do_not_share(self):
        spec = dict(operation='sum', children=[{'input':0}], phi=[{'G':3, 'share':'p'}])
        a, b = self.builder.build(spec), self.builder.build(spec)
        self.assertIsNot(a.phi[0], b.phi[0])

    def test_conflicting_shared_definitions_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.build(dict(operation='sum', children=[{'input':0},{'input':1}],
                phi=[{'G':3,'share':'p'},{'G':9,'share':'p'}]))

    def test_shared_grid_pools_contexts(self):
        data = torch.stack([torch.linspace(0,1,50),torch.linspace(10,20,50)],-1)
        model = self.builder.build(dict(operation='sum',children=[{'input':0},{'input':1}],
            phi=[{'G':5,'share':'p'},{'G':5,'share':'p'}]),train_inputs=data)
        low,high = torch.quantile(data.reshape(-1),torch.tensor([.01,.99]))
        torch.testing.assert_close(model.phi[0].basis.centers,torch.linspace(low,high,5))
        self.assertTrue(torch.equal(model(data),data.sum(-1,keepdim=True)))

    def test_fixed_range_not_recalibrated(self):
        model = self.builder.build(dict(operation='sum', children=[{'input':0}],
            phi=[{'G':3,'range':[-2,2]}]),train_inputs=self.x)
        self.assertTrue(torch.equal(model.phi[0].basis.centers,torch.tensor([-2.,0.,2.])))

    def test_input_and_schema_validation(self):
        for index in (-1,2,True):
            with self.assertRaises(ValueError):self.builder.input(index)
        with self.assertRaises(ValueError):self.builder.affine_route([1.])
        with self.assertRaises(ValueError):
            self.builder.build(dict(operation='sum',children=[{'input':0}],phi=[3],unexpected=True))
        with self.assertRaises(ValueError):
            self.builder.build(dict(operation='sum',children=[{'input':0}],phi=[]))
        with self.assertRaises(ValueError):
            self.builder.build(dict(operation='sum',children=[{'input':0}],phi=[{'range':[0,1,2]}]))

    def test_builder_preserves_spec_and_rng(self):
        spec = dict(operation='sum',children=[{'affine':[1.,1.]}],phi=[5])
        before = copy.deepcopy(spec)
        rng = torch.random.get_rng_state().clone()
        self.builder.build(spec,train_inputs=self.x)
        self.assertEqual(spec,before)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
