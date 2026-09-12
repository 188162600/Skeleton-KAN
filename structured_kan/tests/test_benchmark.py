"""Remote regression checks for the production entry point."""
import json
from pathlib import Path
import tempfile
import unittest
import torch

from ..dataset.fresh_sampling import generate_row,sampling_seed
from ..dataset.fresh_loader import load_seed_data
from ..model import StructuredKANBuilder
from ..model.initialization import perturb_trainable_
from ..model.topology_spec import finite_topology_spec
from ..model.GaussianBasis import GaussianBasis,default_width_ratio
from ..structure_builder.complexity_primitives import score
from ..optimizer.seed_batch_lbfgs import minimize


class BenchmarkTests(unittest.TestCase):
    def test_complexity_rules(self):
        for expression,g in [('z',3),('cos(z)',9),('sin(z)',9),('cos(z)**2',9),
            ('sin(z)**2',9),('1/z',15),('z**(-2)',15),('3/(2*z+1)**2+4',15),
            ('log(z)',9),('sqrt(z)',9)]:
            self.assertEqual(score(dict(primitive='symbolic',symbolic_expression=expression))[0],g)

    def test_fresh_thirty_streams(self):
        path=Path(__file__).resolve().parents[1]/'data/equation_set/eval80/symbolic_search20.json'
        row=json.loads(path.read_text())[0]
        with tempfile.TemporaryDirectory() as directory:
            report=generate_row((row,directory,16));self.assertEqual(report['status'],'complete',report)
            samples,receipt=load_seed_data(row,directory,points=16)
            self.assertEqual(len(samples),10);self.assertEqual(receipt['sampling_sets'],30)
            again=generate_row((row,directory,16));self.assertEqual(report,again)
            self.assertEqual(len({sampling_seed(row['case_id'],seed,split) for seed in range(421,431)
                                  for split in ('train','validation','test')}),30)

    def test_finite_dynamic_models(self):
        tree=dict(operator='+',raw_arity=1,children=[dict(operator='*',raw_arity=2,children=[])])
        budgets={f'phi_{i}':g for i,g in enumerate([3,9,15,3,9])}
        spec=finite_topology_spec(tree,3,per_map_G=budgets,output_G=9,off_diagonal=0)
        x=torch.randn(32,3,dtype=torch.float64);models=[]
        for seed in (421,422):
            model=StructuredKANBuilder(3).build(spec,train_inputs=x)
            self.assertEqual(sum(p.numel() for p in model.parameters()),3*4+sum(g+2 for g in budgets.values()))
            before={n:b.clone() for n,b in model.named_buffers()}
            perturb_trainable_(model,seed)
            self.assertTrue(all(torch.equal(b,before[n]) for n,b in model.named_buffers()))
            model(x).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            models.append(model)
        self.assertFalse(torch.equal(models[0](x),models[1](x)))
        for module in models[0].modules():
            if isinstance(module,GaussianBasis):
                spacing=(module.centers[-1]-module.centers[0])/(module.G-1)
                self.assertAlmostEqual(float(module.width),max(default_width_ratio(module.G)*float(spacing),.001),12)

    def test_progress_is_observational(self):
        x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        def gv(x,indices):return 2*x,x.square().sum(1)
        def value(x):return x.square().sum(1)
        first=minimize(x,gv,value,outer_steps=2)
        seen=[];second=minimize(x,gv,value,outer_steps=2,progress=lambda n,*args:seen.append(n))
        self.assertTrue(torch.equal(first.parameters,second.parameters));self.assertEqual(seen,[1,2])


if __name__=='__main__':unittest.main()
