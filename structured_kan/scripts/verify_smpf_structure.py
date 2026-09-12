"""Remote-only materialization, formula, replay and gradient checks; not a benchmark fit."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path


def main():
    import torch
    from ..structure_builder.smpf import architecture_sweep,layout,parameter_count,specification
    from ..model.SMPFStructure import build_model
    from ..model.StructuredKAN import StructuredKAN
    from ..model.KAN import KAN
    from ..model.StructuredKANBuilder import StructuredKANBuilder, _Coordinate, _AffineRoute
    from ..model.MLP import parameter_hash
    from .benchmark import save
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-mode',choices=['full_affine','coordinate'],default='coordinate');args=parser.parse_args()
    architectures=architecture_sweep(input_mode=args.input_mode)
    report=dict(status='testing',started=time.time(),checks=[])
    def record(**values):report['checks'].append(values);save(args.output,report)
    try:
        torch.set_num_threads(1)
        checked=0
        for d in range(1,11):
            x=torch.linspace(-1.6,1.7,32*d,dtype=torch.float64).reshape(32,d)
            for arch in architectures:
                model=build_model(d,arch,train_inputs=x,seed=421)
                L=arch['hidden_sums'];A=sum(map(len,layout(d,arch)))
                assert model.parameter_count()==parameter_count(d,arch)
                assert len([m for m in model.modules() if isinstance(m,StructuredKAN)])==L+1
                assert len([m for m in model.modules() if isinstance(m,KAN)])==A+L
                assert all(m.operation=='sum' for m in model.modules() if isinstance(m,StructuredKAN))
                # No shared route or phi parameters and no post-root phi.
                assert len(list(model.named_parameters(remove_duplicate=False)))==len(list(model.named_parameters()))
                direct=sum(outer(sum(edge(route(x)) for route,edge in zip(branch.branches,branch.phi)))
                           for branch,outer in zip(model.branches,model.phi))
                torch.testing.assert_close(model(x),direct,rtol=1e-12,atol=1e-12)
                if args.input_mode=='coordinate':
                    assert not any(isinstance(m,_AffineRoute) for m in model.modules())
                    assert all(name.rsplit('.',1)[-1] in ('a','b','coefficients') for name,_ in model.named_parameters())
                    # Independent equation evaluation: do not reuse route.forward.
                    direct_coordinate=sum(outer(sum(edge(x[:,j:j+1]) for edge,j in zip(branch.phi,coords)))
                        for branch,outer,coords in zip(model.branches,model.phi,layout(d,arch)))
                    torch.testing.assert_close(model(x),direct_coordinate,rtol=1e-12,atol=1e-12)
                    for branch,coords in zip(model.branches,layout(d,arch)):
                        assert all(isinstance(leaf,_Coordinate) and leaf.index==j for leaf,j in zip(branch.branches,coords))
                        other=sorted(set(range(d))-set(coords))
                        changed=x.clone();changed[:,other]+=7.25
                        torch.testing.assert_close(branch(x),branch(changed),rtol=0,atol=0)
                rebuilt=StructuredKANBuilder(d).build(specification(d,arch),train_inputs=x)
                rebuilt.load_state_dict(model.state_dict())
                torch.testing.assert_close(rebuilt(x),model(x),rtol=0,atol=0)
                gradients=torch.autograd.grad(model(x).square().mean(),tuple(model.parameters()))
                assert all(torch.isfinite(g).all() for g in gradients)
                checked+=1
        record(check='all_materializations_formula_params_replay_gradients',passed=True,models=checked,dimensions=list(range(1,11)))
        if args.input_mode=='coordinate':
            record(check='coordinate_leaves_no_dense_affine_and_exact_branch_input_isolation',passed=True,models=checked)
        assert torch.cuda.is_available()
        x=torch.linspace(-1.2,1.4,48,dtype=torch.float64,device='cuda').reshape(16,3)
        models=[build_model(3,architectures[0],train_inputs=x,seed=s,device='cuda') for s in range(421,431)]
        assert len({parameter_hash(m) for m in models})==10
        ps,bs=torch.func.stack_module_state(models);template=copy.deepcopy(models[0])
        inputs=torch.stack([x+.001*i for i in range(10)])
        def loss(p,b,z):return torch.func.functional_call(template,(p,b),(z,)).square().mean()
        batched=torch.vmap(torch.func.grad_and_value(loss))
        gradients,values=batched(ps,bs,inputs)
        for i in range(10):
            g,v=torch.func.grad_and_value(loss)({n:t[i] for n,t in ps.items()},{n:t[i] for n,t in bs.items()},inputs[i])
            torch.testing.assert_close(v,values[i],rtol=1e-10,atol=1e-10)
            for name in g:torch.testing.assert_close(g[name],gradients[name][i],rtol=1e-9,atol=1e-9)
        compiled=torch.compile(batched,mode='default',fullgraph=True)
        cg,cv=compiled(ps,bs,inputs)
        torch.testing.assert_close(cv,values,rtol=1e-8,atol=1e-9)
        for name in gradients:torch.testing.assert_close(cg[name],gradients[name],rtol=1e-8,atol=1e-9)
        record(check='ten_seed_independence_eager_compiled_scalar_gradients',passed=True)
        report.update(status='passed',seconds=time.time()-report['started'])
    except BaseException as error:
        import traceback
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:save(args.output,report)
    print(json.dumps(report))


if __name__=='__main__':main()
