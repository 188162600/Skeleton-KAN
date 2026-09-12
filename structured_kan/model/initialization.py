"""Reproducible small perturbations to trainable parameters only."""
import hashlib
import torch


def perturb_trainable_(model,seed,std=.001):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    digest=hashlib.sha256()
    with torch.no_grad():
        for name,parameter in model.named_parameters():
            if not parameter.requires_grad:continue
            noise=torch.randn(parameter.shape,dtype=parameter.dtype,generator=generator)
            parameter.add_(noise.to(parameter.device)*std)
            digest.update(name.encode());digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
