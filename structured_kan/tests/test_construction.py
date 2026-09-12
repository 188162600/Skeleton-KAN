"""Small source-side regressions; no full benchmark training."""
import json
from pathlib import Path
import tempfile
import unittest

from structured_kan.structure_builder.cardinality import next_count, select_representatives
from structured_kan.structure_builder.native_trees import ExprTree, original_tree, ordered_distance
from structured_kan.structure_builder.terms import Term, term_from_spec

PROJECT=Path(__file__).resolve().parents[1]


class ConstructionTest(unittest.TestCase):
    def test_compound_noninputs_are_constant(self):
        tree=original_tree(dict(analysis_expression='x*sqrt(a*a+b*b)',variables=['x']))
        self.assertEqual(tree.name,'prod')
        self.assertEqual(sum(c.name.startswith('constant:') for c in tree.children),1)
        self.assertEqual(sum(c.name=='input:0' for c in tree.children),1)

    def test_named_pi_can_be_input(self):
        tree=original_tree(dict(analysis_expression='sin(pi)+E',variables=['pi']))
        names=[]
        def visit(t):
            names.append(t.name)
            for c in t.children:visit(c)
        visit(tree)
        self.assertIn('input:0',names)
        self.assertIn('constant:E',names)

    def test_unary_colour_and_arity(self):
        self.assertEqual(Term('*',True,(),1),Term('+',True,(),1))
        self.assertNotEqual(Term('*',True,(),2),Term('+',True,(),2))
        self.assertNotEqual(Term('+',True,(),1),Term('+',True,(),2))

    def test_ordered_edit_distance(self):
        a=ExprTree('sum',[ExprTree('x',[]),ExprTree('y',[])])
        b=ExprTree('prod',[ExprTree('x',[]),ExprTree('y',[])])
        self.assertEqual(ordered_distance(a,a),0)
        self.assertEqual(ordered_distance(a,b),1)
        self.assertEqual(ordered_distance(b,a),1)

    def test_shortfall_not_plus_one(self):
        def result(k,n,start=0):
            return dict(status='succeeded',native_k=k,final_distinct_count=n,
                        representatives=[dict(signature=str(i)) for i in range(start,start+n)])
        first=result(18,15)
        self.assertEqual(next_count([first],18),21)
        second=result(21,19)
        chosen,selected,added,exact=select_representatives([first,second],18)
        self.assertEqual(len(selected),18)
        self.assertEqual(len(added),3)
        self.assertFalse(exact)
        self.assertIsNone(next_count([first,second],18))

    def test_source_only_and_complete_ted(self):
        from structured_kan.structure_builder.build import build_catalogue
        source=PROJECT/'data/equation_set/folds/heldout_symbolic_search/training225.json'
        pool=json.loads(source.read_text())
        unique={}
        for row in pool:
            unique.setdefault(term_from_spec(row['operator_variable_spec']).signature(),row)
            if len(unique)==4:break
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp)
            selected=list(unique.values())
            path=temp/'source.json';path.write_text(json.dumps(selected))
            with self.assertRaises(ValueError):
                build_catalogue(path,temp/'leaked',method='ted',k=2,
                                heldout_domain=selected[0]['source_corpus'],workers=1)
            result=build_catalogue(path,temp/'ok',method='ted',k=2,
                                   heldout_domain='symbolic_search',workers=2)
            self.assertEqual(result['builder_count'],2)
            self.assertEqual(len(result['builders']),2)
            self.assertFalse(result['construction']['existing_catalogue_used'])
            self.assertEqual(result['construction']['native_counts'],[2])
            self.assertTrue((temp/'ok/catalogue.json').is_file())
            with self.assertRaises(FileExistsError):
                build_catalogue(path,temp/'ok',method='ted',k=2,workers=1)
            path.write_text(json.dumps(selected[:1]))
            single=build_catalogue(path,temp/'single',method='ted',k=1,
                                   heldout_domain='symbolic_search',workers=1)
            self.assertEqual(single['builder_count'],1)

    def test_pot_transport_and_barycenter(self):
        import numpy as np
        from structured_kan.structure_builder.fgw import categorical_graphs
        from structured_kan.structure_builder.fgw_pot import transport, fgw_barycenters
        trees=[ExprTree('sum',[ExprTree('x',[]),ExprTree('y',[])]),
               ExprTree('prod',[ExprTree('x',[]),ExprTree('y',[])])]
        graphs,_=categorical_graphs(trees)
        plan,log=transport(graphs[0].values(),graphs[0].C,graphs[1].values(),graphs[1].C)
        np.testing.assert_allclose(plan.sum(0),np.ones(3)/3,atol=1e-10)
        np.testing.assert_allclose(plan.sum(1),np.ones(3)/3,atol=1e-10)
        features,structure,info=fgw_barycenters(N=3,Ys=[g.values() for g in graphs],
            Cs=[g.C for g in graphs],ps=[np.ones(3)/3]*2,lambdas=np.ones(2)/2,alpha=.5,
            init_X=graphs[0].values(),init_C=graphs[0].C)
        self.assertEqual(features.shape,graphs[0].values().shape)
        self.assertTrue(np.isfinite(structure).all())
        self.assertEqual(len(info['T']),2)

    def test_native_acuos2_pairs(self):
        from structured_kan.structure_builder.acuos2 import configure,generalize
        trees=[ExprTree('sum',[ExprTree('x',[]),ExprTree('y',[])]),
               ExprTree('prod',[ExprTree('x',[]),ExprTree('y',[])])]
        test,spec,tokens,metadata=configure(trees,PROJECT/'structure_builder/vendor/acuos2_official_d9214e03')
        x,y=tokens['x'],tokens['y']
        for left,right in [(x,y),(f's({x},{y})',f's({y},{x})'),(f'p({x},e1)',x)]:
            choices,info=generalize(test,spec,left,right)
            self.assertTrue(choices)
            self.assertTrue(info['complete'])
            self.assertTrue(info['verified_generalization'])
        self.assertFalse(metadata['inference_rules_changed'])

    def test_shared_synthesis_uses_common_projector(self):
        from structured_kan.structure_builder import synthesis,incidence
        self.assertIs(synthesis.Term,Term)
        self.assertIs(synthesis.term_from_spec,term_from_spec)
        self.assertIs(incidence.ACU,synthesis)

    def test_v2_source_to_shared_reserve(self):
        from structured_kan.structure_builder.build import build_catalogue
        source=PROJECT/'data/equation_set/folds/heldout_symbolic_search/training225.json'
        pool=json.loads(source.read_text())
        unique={}
        for row in pool:
            unique.setdefault(term_from_spec(row['operator_variable_spec']).signature(),row)
            if len(unique)==4:break
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);path=temp/'source.json'
            path.write_text(json.dumps(list(unique.values())))
            result=build_catalogue(path,temp/'v2',method='v2',k=2,heldout_domain='symbolic_search')
            self.assertEqual(result['builder_count'],2)
            self.assertEqual(result['construction']['reserve_quantile'],.95)
            self.assertFalse(result['construction']['heldout_data_supplied'])
            for builder in result['builders']:
                self.assertIn('__shared_pool__',builder['raw_incidence_rule']['port_capacity_by_node'])


if __name__=='__main__':unittest.main()
