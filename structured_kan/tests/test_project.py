import copy
import unittest

import torch
from torch import nn

from structured_kan.dataset import RegressionData, Split, sample_box, regression_metrics, geometric_mean_nmse
from structured_kan.optimizer import minimize, fit_seed_batch
from structured_kan.structure_builder import Catalogue
from structured_kan.model import StructuredKANBuilder


class ProjectTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)

    def test_sampling_reproducible_and_independent(self):
        def sample(seed):return sample_box(lambda x:x.sum(-1),[[-1,1],[0,2]],equation='test',seed=seed,points=32)
        a,b,c=sample(421),sample(421),sample(422)
        self.assertTrue(torch.equal(a.train.x,b.train.x))
        self.assertFalse(torch.equal(a.train.x,c.train.x))
        self.assertFalse(torch.equal(a.train.x,a.validation.x))
        self.assertFalse(torch.equal(a.validation.x,a.test.x))

    def test_train_only_standardization(self):
        train=Split(torch.tensor([[0.],[2.]]),torch.tensor([[1.],[3.]]))
        validation=Split(torch.tensor([[100.],[102.]]),torch.tensor([[101.],[103.]]))
        data=RegressionData(train,validation,validation,'test',421)
        normalized,statistics=data.standardize()
        self.assertEqual(float(statistics['x_mean']),1.)
        self.assertEqual(float(statistics['y_mean']),2.)
        self.assertTrue(torch.equal(normalized.validation.x,torch.tensor([[99.],[101.]])))

    def test_metrics(self):
        result=regression_metrics(torch.tensor([1.,3.]),torch.tensor([0.,2.]))
        self.assertEqual(result,dict(mse=1.,nmse=1.))
        self.assertAlmostEqual(geometric_mean_nmse([0.,1e-8]),1e-9)

    def test_catalogue_rejects_duplicates(self):
        row=dict(id='a',structure=dict(operation='sum',children=[{'input':0}],phi=[3]))
        with self.assertRaises(ValueError):Catalogue([row,row])
        catalogue=Catalogue([row])
        self.assertEqual(catalogue.sha256,Catalogue([copy.deepcopy(row)]).sha256)
        self.assertEqual(catalogue.materialize(1)['a'].parameter_count(),5)

    def test_optimizer_independent_quadratics(self):
        target=torch.tensor([[1.,2.],[-3.,4.]])
        initial=torch.tensor([[.3,-.4],[.5,-.6]])
        def gradient(rows,indices):
            delta=rows-target.index_select(0,indices)
            return 2*delta,delta.square().sum(1)
        value=lambda rows:(rows-target).square().sum(1)
        result=minimize(initial,gradient,value,outer_steps=2)
        torch.testing.assert_close(result.parameters,target,rtol=0,atol=1e-12)
        self.assertEqual(len(result.inner_iterations),2)

    def test_generic_model_optimizer_adapter(self):
        x=torch.linspace(-1,1,16).unsqueeze(-1)
        models=[nn.Linear(1,1),nn.Linear(1,1)]
        datasets=[]
        for index in range(2):
            y=(index+2)*x+(index-.5)
            split=Split(x,y)
            datasets.append(RegressionData(split,split,split,'unit_test_only',index))
        fit_seed_batch(models,datasets,outer_steps=2)
        for model,data in zip(models,datasets):
            self.assertLess(float((model(x)-data.train.y).square().mean().detach()),1e-15)

    def test_optimizer_preserves_shared_modules(self):
        builder=StructuredKANBuilder(1)
        spec=dict(operation='sum',children=[{'input':0},{'input':0}],
                  phi=[{'G':3,'share':'shared'},{'G':3,'share':'shared'}])
        models=[builder.build(spec) for _ in range(2)]
        x=torch.linspace(-1,1,16).unsqueeze(-1)
        data=[]
        for index in range(2):
            split=Split(x,(index+3)*x+0.2)
            data.append(RegressionData(split,split,split,'unit_test_only',index))
        old_buffers=[{n:b.clone() for n,b in m.named_buffers()} for m in models]
        fit_seed_batch(models,data,outer_steps=2)
        for index,(model,values) in enumerate(zip(models,data)):
            self.assertIs(model.phi[0],model.phi[1])
            self.assertEqual(model.parameter_count(),5)
            self.assertLess(float((model(x)-values.train.y).square().mean().detach()),1e-15)
            for name,buffer in model.named_buffers():
                self.assertTrue(torch.equal(buffer,old_buffers[index][name]))


if __name__=='__main__':unittest.main()
