"""Remote fidelity and integration checks for Fourier-MFN."""
import importlib.util
import json
from pathlib import Path
import hashlib
import unittest
import torch
from ..model.FourierMFN import FourierMFN, architecture_sweep, parameter_count
from ..model.MLP import parameter_hash
from ..model.neural_baselines import build_model
from ..scripts.neural_benchmark import inspect_config


class FourierTests(unittest.TestCase):
    def test_official_equivalence(self):
        root=Path(__file__).resolve().parents[1]/'vendor/fourier_mfn_official'
        provenance=json.loads((root/'provenance.json').read_text())
        self.assertEqual(provenance['commit'],'58c3c4dd5908bce9d5c3b077926cb588cc6052da')
        self.assertEqual(hashlib.sha256((root/'mfn.py').read_bytes()).hexdigest(),provenance['files']['mfn.py']['sha256'])
        spec=importlib.util.spec_from_file_location('official_mfn',root/'mfn.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        old=torch.get_default_dtype();torch.set_default_dtype(torch.float64)
        try:
            for d in (2,3,10):
                for arch in architecture_sweep():
                    for seed in (421,430):
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(seed)
                            reference=native.FourierNet(d,arch['hidden_width'],1,n_layers=arch['product_layers'],
                                input_scale=arch['input_scale'],weight_scale=1.0,bias=True,output_act=False)
                        model=build_model('fourier_mfn',d,arch,seed=seed)
                        self.assertEqual(set(reference.state_dict()),set(model.state_dict()))
                        for name,value in reference.state_dict().items():
                            self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
                        x=torch.linspace(-2,2,7*d).reshape(7,d)
                        self.assertTrue(torch.equal(reference(x),model(x)))
                        reference(x).square().mean().backward();model(x).square().mean().backward()
                        for (a,p),(b,q) in zip(reference.named_parameters(),model.named_parameters()):
                            self.assertEqual(a,b);self.assertTrue(torch.equal(p.grad,q.grad),a)
                        self.assertEqual(sum(p.numel() for p in model.parameters()),parameter_count(d,arch['product_layers'],arch['hidden_width']))
        finally:
            torch.set_default_dtype(old)

    def test_independent_seed_initialization(self):
        models=[FourierMFN(3,2,32,1.0,seed=s) for s in range(421,431)]
        self.assertEqual(len({parameter_hash(m) for m in models}),10)
        self.assertEqual(parameter_hash(models[0]),parameter_hash(FourierMFN(3,2,32,1.0,seed=421)))

    def test_vmap_gradients(self):
        models=[FourierMFN(2,2,4,4.0,seed=s) for s in (421,422)]
        params,buffers=torch.func.stack_module_state(models)
        x=torch.randn(2,13,2,dtype=torch.float64);y=torch.randn(2,13,1,dtype=torch.float64)
        def loss(p,b,x,y):return (torch.func.functional_call(models[0],(p,b),(x,))-y).square().mean()
        grads=torch.vmap(torch.func.grad(loss))(params,buffers,x,y)
        for i,model in enumerate(models):
            (model(x[i])-y[i]).square().mean().backward()
            for name,p in model.named_parameters():
                torch.testing.assert_close(grads[name][i],p.grad,atol=1e-12,rtol=1e-12)

    def test_corrected_depths_and_configs(self):
        _,config,rows=inspect_config('fourier_mfn18_domain80.json')
        self.assertEqual(len(rows),80)
        self.assertEqual(len(config['architectures']),18)
        self.assertEqual({a['product_layers'] for a in config['architectures']},{1,2,4})
        self.assertEqual({a['hidden_width'] for a in config['architectures']},{32,64,128})
        self.assertEqual({a['input_scale'] for a in config['architectures']},{1.0,4.0})


if __name__=='__main__':unittest.main()
