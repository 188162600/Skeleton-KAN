"""Analytic PDE-10: physical-coordinate residuals and seeded collocation data.

Exact interior solutions are evaluation labels only. Training sees the PDE
forcing and prescribed boundary/initial values, never interior solution labels.
All scientific evaluation/sampling is deliberately remote-only.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

MANIFEST=Path(__file__).resolve().parents[1]/'data/equation_set/pde10/problems.json'

def problems():
    return json.loads(MANIFEST.read_text())['problems']

def derivative(function,x,axis,order=1):
    """Pointwise reverse-mode derivatives; models never couple sample rows."""
    if order==0:return function(x)
    gradient=torch.func.grad(lambda z:derivative(function,z,axis,order-1).sum())(x)
    return gradient[...,axis:axis+1]

def forward_derivative(function,x,axis,order=1):
    """Independent forward-mode reference, without an N-by-N Jacobian."""
    if order==0:return function(x)
    tangent=torch.zeros_like(x)
    # No in-place writes to vmapped tensors: a constant coordinate mask.
    mask=(torch.arange(x.shape[-1],device=x.device)==axis).to(x.dtype)
    tangent=tangent+mask
    return torch.func.jvp(lambda z:forward_derivative(function,z,axis,order-1),(x,),(tangent,))[1]

def exact(problem,x):
    name=problem['id'];a=x[...,0:1];b=x[...,1:2];pi=torch.pi
    if name=='advection':return torch.sin(pi*(a-b))
    if name=='burgers':return .5-.5*torch.tanh(2.5*(a-.5*b))
    if name=='fisher_kpp':return torch.sigmoid(-(a/(.6**.5)-5*b/6)).square()
    if name=='anisotropic_heat':
        t=x[...,2:3]
        return torch.exp(-.9*pi*pi*t)*torch.sin(pi*a)*torch.sin(2*pi*b)
    if name=='manufactured_heat':
        return torch.exp(-x[...,2:3])*torch.sin(pi*a*b)
    if name=='wave':return torch.sin(pi*a)*torch.cos(2*pi*b)+.5*torch.sin(4*pi*a)*torch.cos(8*pi*b)
    if name=='laplace':return torch.sin(pi*a)*torch.sinh(pi*b)/torch.sinh(x.new_tensor(pi))
    if name=='helmholtz':return torch.sin(4*pi*a)*torch.sin(4*pi*b)
    if name=='allen_cahn':return torch.sigmoid(-(a*(5**.5)-1.5*b))
    if name=='kdv':return .5/torch.cosh(.5*(a-b)).square()
    raise KeyError(name)

def forcing(problem,x):
    if problem['id']=='manufactured_heat':
        q=x[...,0:1].square()+x[...,1:2].square()
        return (torch.pi**2*(1+q)*q-1)*exact(problem,x)
    if problem['id']=='helmholtz':return (4*torch.pi)**2*exact(problem,x)
    return torch.zeros_like(x[...,0:1])

def residual(problem,function,x,source):
    name=problem['id']
    differentiate=forward_derivative if os.environ.get('STRUCTURED_KAN_PDE_FORWARD_AD')=='1' else derivative
    d=lambda axis,order=1:differentiate(function,x,axis,order)
    if name=='advection':return d(1)+d(0)
    if name=='burgers':return d(1)+function(x)*d(0)-.1*d(0,2)
    if name=='fisher_kpp':
        u=function(x);return d(1)-.1*d(0,2)-u*(1-u)
    if name=='anisotropic_heat':return d(2)-.1*d(0,2)-.2*d(1,2)
    if name=='manufactured_heat':
        coefficient=1+x[...,0:1].square()+x[...,1:2].square()
        return d(2)-coefficient*(d(0,2)+d(1,2))-source
    if name=='wave':return d(1,2)-4*d(0,2)
    if name=='laplace':return d(0,2)+d(1,2)
    if name=='helmholtz':return -d(0,2)-d(1,2)-(4*torch.pi)**2*function(x)-source
    if name=='allen_cahn':
        u=function(x);return d(1)-.1*d(0,2)-u+u.pow(3)
    if name=='kdv':return d(1)+6*function(x)*d(0)+d(0,3)
    raise KeyError(name)

def stream_seed(problem,seed,split):
    key=f"pde10-v1|{problem['id']}|{seed}|{split}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)

def sample(problem,seed,*,points=10000,boundary_points=2000,initial_points=2000,device='cpu'):
    streams={}
    bounds=torch.tensor(problem['bounds'],dtype=torch.float64)
    def box(n,label):
        value=stream_seed(problem,seed,label);streams[label]=value
        rng=torch.Generator().manual_seed(value)
        return bounds[:,0]+torch.rand(n,len(bounds),generator=rng,dtype=torch.float64)*(bounds[:,1]-bounds[:,0])
    x=box(points,'train_interior');xb=box(boundary_points,'train_boundary')
    faces=problem['dirichlet_faces']
    for index,(axis,side) in enumerate(faces):xb[index::len(faces),axis]=bounds[axis,side]
    yb=exact(problem,xb)
    time_axis=problem.get('time_axis')
    xi=box(initial_points if time_axis is not None else 0,'train_initial')
    if time_axis is not None:xi[:,time_axis]=bounds[time_axis,0]
    yi=exact(problem,xi)
    dc=problem.get('derivative_condition')
    xd=box(initial_points if dc else 0,'train_derivative_condition')
    if dc:
        xd[:,dc['face_axis']]=bounds[dc['face_axis'],dc['side']]
        yd=derivative(lambda z:exact(problem,z),xd,dc['derivative_axis'])
    else:yd=xd[:,:1]
    values=torch.cat([yb,yi]);ym=values.mean();ys=values.std(unbiased=False)
    # Homogeneous traces (e.g. Helmholtz) do not identify an output scale.
    # A fixed unit scale avoids learning it from forbidden interior labels.
    if float(ys)<1e-6:ym=torch.zeros_like(ym);ys=torch.ones_like(ys)
    tensors=dict(interior=x,forcing=forcing(problem,x),boundary=xb,boundary_y=yb,
        initial=xi,initial_y=yi,derivative_x=xd,derivative_y=yd,
        x_mean=x.mean(0),x_std=x.std(0,unbiased=False),y_mean=ym,y_std=ys)
    evaluation={}
    for split in ('validation','test'):
        xx=box(points,split);evaluation[split]=dict(x=xx,y=exact(problem,xx))
    receipt=dict(problem=problem['id'],seed=seed,stream_seeds=streams,
        train_interior_points=points,boundary_points=len(xb),initial_points=len(xi),derivative_points=len(xd),
        validation_points=points,test_points=points,normalization='input: train interior; output: train boundary and initial values only',
        x_mean=tensors['x_mean'].tolist(),x_std=tensors['x_std'].tolist(),y_mean=float(ym),y_std=float(ys),
        test_target_min=float(evaluation['test']['y'].min()),test_target_max=float(evaluation['test']['y'].max()),
        training_interior_solution_labels_used=False)
    assert len(set(streams.values()))==len(streams)
    assert all(bool(torch.isfinite(t).all()) for t in tensors.values())
    assert bool((tensors['x_std']>0).all())
    return ({k:v.to(device) for k,v in tensors.items()},
            {s:{k:v.to(device) for k,v in q.items()} for s,q in evaluation.items()},receipt)

def training_loss(problem,predict,data):
    def physical(x):return predict((x-data['x_mean'])/data['x_std'])*data['y_std']+data['y_mean']
    scale=data['y_std']
    pde=(residual(problem,physical,data['interior'],data['forcing'])/scale).square().mean()
    boundary=((physical(data['boundary'])-data['boundary_y'])/scale).square().mean()
    loss=.01*pde+boundary
    if problem.get('time_axis') is not None:
        loss=loss+((physical(data['initial'])-data['initial_y'])/scale).square().mean()
    dc=problem.get('derivative_condition')
    if dc:
        differentiate=forward_derivative if os.environ.get('STRUCTURED_KAN_PDE_FORWARD_AD')=='1' else derivative
        loss=loss+((differentiate(physical,data['derivative_x'],dc['derivative_axis'])-data['derivative_y'])/scale).square().mean()
    return loss
