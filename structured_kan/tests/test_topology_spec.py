import unittest

import torch

from structured_kan.model import StructuredKANBuilder
from structured_kan.model.topology_spec import finite_topology_spec


class FiniteTopologyTest(unittest.TestCase):
    def test_full_affine_routes_and_incoming_maps(self):
        tree=dict(operator='*',raw_arity=2,raw_attachment=True,children=[])
        model=StructuredKANBuilder(2).build_finite_topology(tree,G=3)
        self.assertEqual(model.parameter_count(),2*(2+1)+2*(3+2))
        x=torch.tensor([[2.,3.]],dtype=torch.float64)
        torch.testing.assert_close(model(x),torch.tensor([[2.03*3.02]],dtype=torch.float64))

    def test_output_phi_is_only_explicit(self):
        tree=dict(operator='+',raw_arity=1,children=[])
        builder=StructuredKANBuilder(2)
        self.assertEqual(builder.build_finite_topology(tree,G=3).parameter_count(),8)
        self.assertEqual(builder.build_finite_topology(tree,G=3,output_G=5).parameter_count(),15)

    def test_per_map_budgets_match_postorder(self):
        tree=dict(operator='+',raw_arity=1,children=[dict(operator='*',raw_arity=2,children=[])])
        budgets={f'phi_{i}':g for i,g in enumerate([3,5,9,15,3])}
        spec=finite_topology_spec(tree,2,per_map_G=budgets,output_G=3)
        self.assertEqual(spec['phi'],[3])
        self.assertEqual(spec['children'][0]['phi'],[9,15])
        self.assertEqual(spec['children'][0]['children'][1]['phi'],[3,5])
        model=StructuredKANBuilder(2).build(spec)
        self.assertEqual(model.parameter_count(),3*3+sum(g+2 for g in budgets.values()))
        with self.assertRaises((ValueError,KeyError)):
            finite_topology_spec(tree,2,per_map_G={'phi_0':3})

    def test_shared_rules_not_silently_discarded(self):
        tree=dict(operator='+',raw_arity=2,children=[],raw_incidence_rule={})
        with self.assertRaises(ValueError):finite_topology_spec(tree,2)


if __name__=='__main__':unittest.main()
