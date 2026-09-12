"""Connect a same-shape model batch to the model-independent L-BFGS core."""
import copy

import torch

from .seed_batch_lbfgs import minimize


def fit_seed_batch(models, datasets, *, outer_steps=50, compile_mode=None, progress=None):
    """Fit independent seeds jointly, with full training sets and frozen buffers.

    Models are updated in place to each seed's best training iterate. Data
    should already be standardized using its own training statistics.
    """
    if not models or len(models) != len(datasets):
        raise ValueError("one model is required per seed dataset")
    params, buffers = torch.func.stack_module_state(models)
    template = copy.deepcopy(models[0])
    # Swap each actual attribute once, even if the same module occurs on
    # multiple routes. Expand shared tensors held by *different* modules.
    # functional_call's automatic alias expansion otherwise swaps a repeated
    # module attribute twice and can restore a temporary BatchedTensor.
    canonical = {id(v): n for n, v in template.named_parameters()}
    canonical.update({id(v): n for n, v in template.named_buffers()})
    parameter_aliases, buffer_aliases = {}, {}
    for module_name, module in template.named_modules():
        prefix = module_name + '.' if module_name else ''
        for name, tensor in module.named_parameters(recurse=False):
            parameter_aliases[prefix + name] = canonical[id(tensor)]
        for name, tensor in module.named_buffers(recurse=False):
            buffer_aliases[prefix + name] = canonical[id(tensor)]
    names = [n for n,p in template.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError("there are no trainable parameters")
    frozen = {n:p for n,p in params.items() if n not in names}
    shapes = {n:params[n].shape[1:] for n in names}
    offsets, offset = {}, 0
    for name in names:
        size = params[name][0].numel()
        offsets[name] = (offset,offset+size);offset += size
    initial = torch.cat([params[n].reshape(len(models),-1) for n in names],1).detach()
    device = initial.device
    x = torch.stack([d.train.x.to(device) for d in datasets])
    y = torch.stack([d.train.y.to(device) for d in datasets])

    def one_loss(row, fixed, buf, data_x, data_y):
        weights = dict(fixed)
        weights.update({n:row[a:b].view(shapes[n]) for n,(a,b) in offsets.items()})
        weights = {alias: weights[name] for alias, name in parameter_aliases.items()}
        expanded_buffers = {alias: buf[name] for alias, name in buffer_aliases.items()}
        prediction = torch.func.functional_call(template,(weights,expanded_buffers),(data_x,),tie_weights=False)
        return (prediction-data_y).square().mean()

    gradient = torch.vmap(torch.func.grad_and_value(one_loss))
    loss = torch.vmap(one_loss)
    if compile_mode is not None:
        gradient = torch.compile(gradient,mode=compile_mode,fullgraph=True,dynamic=False)
        loss = torch.compile(loss,mode=compile_mode,fullgraph=True,dynamic=False)

    def value_gradient(rows, indices):
        fixed = {n:p.index_select(0,indices) for n,p in frozen.items()}
        buf = {n:p.index_select(0,indices) for n,p in buffers.items()}
        return gradient(rows,fixed,buf,x.index_select(0,indices),y.index_select(0,indices))

    result = minimize(initial,value_gradient,
        lambda rows:loss(rows,frozen,buffers,x,y).detach().clone(),outer_steps=outer_steps,progress=progress)
    with torch.no_grad():
        for index,model in enumerate(models):
            by_name = dict(model.named_parameters())
            for name,(a,b) in offsets.items():
                by_name[name].copy_(result.parameters[index,a:b].view(shapes[name]))
    return result
