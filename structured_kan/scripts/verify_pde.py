"""Remote PDE residual, gradient, seed-independence and optimizer smoke audit."""
import argparse
import json
import sys
import time
from pathlib import Path

def main():
    import torch
    from ..dataset.pde import problems,exact,forcing,residual,sample,training_loss,derivative,forward_derivative
    from ..optimizer.pde_fit import Objective,fit_pde_seed_batch
    from .pde_benchmark import inspect_config,candidates,make_model
    from .benchmark import save
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    started=time.time();path,c,ps=inspect_config(args.config);assert torch.cuda.is_available()
    report=dict(status='testing',method=c['method'],pid=__import__('os').getpid(),started=started,checks=[])
    def record(**r):report['checks'].append(r);save(args.output,report)
    try:
        if c['method']=='multkan':
            from ..model.MultKAN import native_model,parameter_count
            for dim in (2,3):
                for candidate in candidates(c):
                    options={k:candidate[k] for k in ('first_hidden','second_hidden','grid')}
                    native=native_model(dim,**options,seed=421,optimized=False).to('cuda')
                    fast,_,_=make_model(c,{'coordinates':list(range(dim))},candidate,421,{})
                    fast.network.load_state_dict(native.state_dict())
                    assert sum(q.numel() for q in fast.parameters() if q.requires_grad)==parameter_count(dim,**options)
                    x=torch.linspace(-1.732,1.731,17*dim,device='cuda',dtype=torch.float64).reshape(17,dim)
                    for axis in range(dim):
                        for order in range(4):
                            actual=forward_derivative(fast,x,axis,order)
                            reference=forward_derivative(native,x,axis,order)
                            torch.testing.assert_close(actual,reference,rtol=1e-8,atol=1e-8)
            record(check='all18_native_MultKAN_values_and_derivatives_orders0to3',dimensions=[2,3],passed=True)
        for p in ps:
            d,e,receipt=sample(p,421,device='cuda')
            x=d['interior'];r=residual(p,lambda z:exact(p,z),x,forcing(p,x))
            maximum=float(r.abs().max());assert maximum<1e-10,(p['id'],maximum)
            for axis in range(x.shape[-1]):
                for order in range(1,4 if p['id']=='kdv' else 3):
                    a=derivative(lambda z:exact(p,z),x[:32],axis,order)
                    b=forward_derivative(lambda z:exact(p,z),x[:32],axis,order)
                    assert torch.allclose(a,b,rtol=1e-10,atol=1e-10)
            def normalized(z):return (exact(p,z*d['x_std']+d['x_mean'])-d['y_mean'])/d['y_std']
            loss=float(training_loss(p,normalized,d));assert loss<1e-16,(p['id'],loss)
            # Independent closed-form trace checks for derivative conditions.
            if p['id']=='wave':assert float(d['derivative_y'].abs().max())<1e-12
            if p['id']=='kdv':
                z=.5*(d['derivative_x'][:,:1]-d['derivative_x'][:,1:2])
                expected=-.5*torch.tanh(z)/torch.cosh(z).square()
                assert torch.allclose(d['derivative_y'],expected,rtol=1e-12,atol=1e-12)
            record(check='exact_residual_and_normalization_chain_rule',problem=p['id'],points=10000,max_abs_residual=maximum,
                exact_pinn_loss=loss,test_range=[receipt['test_target_min'],receipt['test_target_max']])
        # All equation-specific differential operators, not just first order.
        for p in ps:
            models=[];data=[];hashes=[]
            for seed in (421,422):
                d,_,_=sample(p,seed,points=16,boundary_points=8,initial_points=8,device='cuda')
                m,_,h=make_model(c,p,candidates(c)[0],seed,d);models.append(m);data.append(d);hashes.append(h)
            assert hashes[0]!=hashes[1] and not torch.equal(data[0]['interior'],data[1]['interior'])
            objective=Objective(models,data,p);rows=objective.initial;indices=torch.arange(2,device='cuda')
            g,v=objective.value_gradient(rows,indices)
            assert bool(torch.isfinite(g).all() & torch.isfinite(v).all())
            max_diff=0.
            for i in range(2):
                fixed={n:t[i] for n,t in objective.fixed.items()};buf={n:t[i] for n,t in objective.buffers.items()}
                one={n:t[i] for n,t in objective.data.items()}
                gi,vi=torch.func.grad_and_value(objective.one_loss)(rows[i],fixed,buf,one)
                assert torch.allclose(gi,g[i],rtol=1e-8,atol=1e-9) and torch.allclose(vi,v[i],rtol=1e-10,atol=1e-10)
                max_diff=max(max_diff,float((gi-g[i]).abs().max()))
            # Finite-difference check of a nonzero parameter gradient.
            k=int(g[0].abs().argmax());eps=1e-5;delta=torch.zeros_like(rows);delta[0,k]=eps
            fd=(objective.value(rows+delta)[0]-objective.value(rows-delta)[0])/(2*eps)
            assert torch.allclose(fd,g[0,k],rtol=2e-4,atol=1e-5),(p['id'],float(fd),float(g[0,k]))
            record(check='seed_batch_scalar_gradient_and_finite_difference',problem=p['id'],max_absolute_batch_difference=max_diff)
        # Compiled higher derivatives must agree with eager gradients and values.
        p=next(p for p in ps if p['id']=='kdv');models=[];data=[]
        def contains_product(tree):return tree['operator']=='*' or any(contains_product(child) for child in tree['children'])
        compiled_candidate=candidates(c)[0]
        if c['method'] not in ('mlp','multkan','fourier_mfn','smpf_structure'):
            compiled_candidate=next(b for b in candidates(c) if contains_product(b['synthetic_operator_tree']))
        elif c['method']=='multkan':
            compiled_candidate=candidates(c)[8]
        for seed in (421,422):
            d,_,_=sample(p,seed,points=16,boundary_points=8,initial_points=8,device='cuda')
            m,_,_=make_model(c,p,compiled_candidate,seed,d);models.append(m);data.append(d)
        if c['method'] not in ('mlp','multkan','fourier_mfn','smpf_structure'):
            import copy
            from ..model.StructuredKAN import StructuredKAN
            reference=copy.deepcopy(models[0])
            for node in reference.modules():
                if isinstance(node,StructuredKAN):node.product_reduction='native'
            x=data[0]['interior']
            assert torch.allclose(reference(x),models[0](x),rtol=1e-12,atol=1e-12)
            for axis in range(x.shape[-1]):
                for order in (1,2,3):
                    assert torch.allclose(derivative(reference,x,axis,order),derivative(models[0],x,axis,order),rtol=1e-8,atol=1e-9)
            record(check='explicit_product_vs_native_forward_and_three_derivative_orders',passed=True)
        objective=Objective(models,data,p);compiled=Objective(models,data,p,compile_mode=c['compile_mode']);idx=torch.arange(2,device='cuda')
        eager=objective.value_gradient(objective.initial,idx);actual=compiled.value_gradient(compiled.initial,idx)
        assert all(torch.allclose(a,b,rtol=1e-8,atol=1e-9) for a,b in zip(eager,actual))
        changed=objective.initial+.002
        eager=objective.value_gradient(changed,idx);actual=compiled.value_gradient(changed,idx)
        assert all(torch.allclose(a,b,rtol=1e-8,atol=1e-9) for a,b in zip(eager,actual))
        assert torch.allclose(objective.value(changed),compiled.value(changed),rtol=1e-8,atol=1e-9)
        # Independent line searches pad active requests to the permanent batch
        # size (see minimize.evaluate_requests); repeat both rows and objectives.
        subset=torch.tensor([1,1],device='cuda')
        eager_subset=objective.value_gradient(changed[subset],subset)
        actual_subset=compiled.value_gradient(changed[subset],subset)
        assert all(torch.allclose(a,b,rtol=1e-8,atol=1e-9) for a,b in zip(eager_subset,actual_subset))
        record(check='compiled_KdV_third_derivative_parameter_gradient',passed=True,
            full_batch_and_padded_active_seed=True,loss_checked=True,
            cudagraphs_override=__import__('os').environ.get('STRUCTURED_KAN_PDE_CUDAGRAPHS'))
        # One outer call is a software smoke test, not a reported scientific fit.
        p=ps[0];data=[];models=[]
        for seed in (421,422):
            d,_,_=sample(p,seed,points=32,boundary_points=16,initial_points=16,device='cuda')
            m,_,_=make_model(c,p,candidates(c)[0],seed,d);data.append(d);models.append(m)
        objective=Objective(models,data,p);before=objective.value(objective.initial)
        result=fit_pde_seed_batch(models,data,p,outer_steps=1,compile_mode=c['compile_mode'])
        assert bool(torch.isfinite(result.loss).all() & (result.loss<=before+1e-9).all())
        record(check='unchanged_seed_batch_lbfgs_smoke_only',outer_calls=1,compiled=True,before=before.tolist(),after=result.loss.tolist())
        report.update(status='passed',seconds=time.time()-started)
    except BaseException as error:
        import traceback
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc(),seconds=time.time()-started);raise
    finally:save(args.output,report)
    print(json.dumps(report))
if __name__=='__main__':main()
