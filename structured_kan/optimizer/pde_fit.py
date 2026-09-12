"""Independent-seed PINN objectives using the unchanged seed-batched L-BFGS."""
import copy
import os
import torch
from .seed_batch_lbfgs import minimize
from ..dataset.pde import training_loss

class Objective:
    def __init__(self,models,data,problem,compile_mode=None):
        params,buffers=torch.func.stack_module_state(models)
        template=copy.deepcopy(models[0]);self.models=models
        canonical={id(v):n for n,v in template.named_parameters()}
        canonical.update({id(v):n for n,v in template.named_buffers()})
        aliases,baliases={},{}
        for mn,module in template.named_modules():
            prefix=mn+'.' if mn else ''
            for n,t in module.named_parameters(recurse=False):aliases[prefix+n]=canonical[id(t)]
            for n,t in module.named_buffers(recurse=False):baliases[prefix+n]=canonical[id(t)]
        names=[n for n,p in template.named_parameters() if p.requires_grad]
        fixed={n:p for n,p in params.items() if n not in names}
        shapes={n:params[n].shape[1:] for n in names};offsets={};offset=0
        for n in names:
            size=params[n][0].numel();offsets[n]=(offset,offset+size);offset+=size
        self.initial=torch.cat([params[n].reshape(len(models),-1) for n in names],1).detach()
        self.offsets,self.shapes=offsets,shapes
        self.data={k:torch.stack([d[k] for d in data]) for k in data[0]}
        self.fixed,self.buffers=fixed,buffers
        def one_loss(row,fixed,buf,batch):
            weights=dict(fixed)
            weights.update({n:row[a:b].view(shapes[n]) for n,(a,b) in offsets.items()})
            weights={a:weights[n] for a,n in aliases.items()}
            expanded={a:buf[n] for a,n in baliases.items()}
            def predict(x):return torch.func.functional_call(template,(weights,expanded),(x,),tie_weights=False)
            return training_loss(problem,predict,batch)
        self.one_loss=one_loss
        self.gradient=torch.vmap(torch.func.grad_and_value(one_loss))
        self.loss=torch.vmap(one_loss)
        if compile_mode is not None:
            if os.environ.get('STRUCTURED_KAN_PDE_REAL_FX')=='1':
                # Exact tensor-graph lowering for nested AD; no finite differences.
                from torch.fx.experimental.proxy_tensor import make_fx
                arguments=(self.initial,self.fixed,self.buffers,self.data)
                self.gradient=make_fx(self.gradient,tracing_mode='real')(*arguments)
                self.loss=make_fx(self.loss,tracing_mode='real')(*arguments)
            compile_kwargs=dict(mode=compile_mode)
            if os.environ.get('STRUCTURED_KAN_PDE_CUDAGRAPHS')=='0':
                # Keep the selected Inductor mode's other options. Nested AD
                # can retain storage wrappers that CUDA graph capture rejects.
                # Disabling capture changes dispatch, not the loss or optimizer.
                options=dict(torch._inductor.list_mode_options(compile_mode))
                options['triton.cudagraphs']=False
                compile_kwargs=dict(options=options)
            self.gradient=torch.compile(self.gradient,fullgraph=True,dynamic=False,**compile_kwargs)
            self.loss=torch.compile(self.loss,fullgraph=True,dynamic=False,**compile_kwargs)

    def value_gradient(self,rows,indices):
        select=lambda d:{n:t.index_select(0,indices) for n,t in d.items()}
        return self.gradient(rows,select(self.fixed),select(self.buffers),select(self.data))

    def value(self,rows):return self.loss(rows,self.fixed,self.buffers,self.data).detach().clone()

    def restore(self,rows):
        with torch.no_grad():
            for i,model in enumerate(self.models):
                params=dict(model.named_parameters())
                for n,(a,b) in self.offsets.items():params[n].copy_(rows[i,a:b].view(self.shapes[n]))

def fit_pde_seed_batch(models,data,problem,*,outer_steps=50,compile_mode=None,progress=None):
    objective=Objective(models,data,problem,compile_mode=compile_mode)
    result=minimize(objective.initial,objective.value_gradient,objective.value,outer_steps=outer_steps,progress=progress)
    objective.restore(result.parameters)
    return result
