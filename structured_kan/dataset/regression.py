"""Independent seeded splits, train-only standardization and regression metrics."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch


@dataclass(frozen=True)
class Split:
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        if self.x.ndim != 2 or self.y.shape != (len(self.x), 1) or not len(self.x):
            raise ValueError("split shapes must be (N,d) and (N,1), with N>0")
        if not bool(torch.isfinite(self.x).all() & torch.isfinite(self.y).all()):
            raise ValueError("a split contains nonfinite data; do not silently filter samples")


@dataclass(frozen=True)
class RegressionData:
    train: Split
    validation: Split
    test: Split
    equation: str
    seed: int

    def standardize(self):
        x_mean, y_mean = self.train.x.mean(0), self.train.y.mean(0)
        x_std, y_std = self.train.x.std(0, unbiased=False), self.train.y.std(0, unbiased=False)
        if not bool((x_std > 0).all() & (y_std > 0).all()):
            raise ValueError("standardization needs nonconstant training coordinates and target")
        def transform(split):
            return Split((split.x-x_mean)/x_std, (split.y-y_mean)/y_std)
        normalized = RegressionData(*(transform(s) for s in (self.train,self.validation,self.test)),
                                    self.equation,self.seed)
        return normalized, dict(x_mean=x_mean,x_std=x_std,y_mean=y_mean,y_std=y_std)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(equation=self.equation,seed=self.seed,
                        splits={k:dict(x=getattr(self,k).x,y=getattr(self,k).y)
                                for k in ('train','validation','test')}),path)

    @classmethod
    def load(cls, path):
        value = torch.load(path,map_location='cpu',weights_only=True)
        return cls(*(Split(**value['splits'][k]) for k in ('train','validation','test')),
                   value['equation'],value['seed'])


def sample_box(function, bounds, *, equation, seed, points=10_000, dtype=torch.float64):
    """Use only for tasks whose declared distribution is uniform on a box.

    This is not a replacement for trajectory/experimental source samplers.
    Each (equation, seed, split) has its own deterministic RNG stream.
    """
    bounds = torch.as_tensor(bounds,dtype=dtype)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not bool(torch.isfinite(bounds).all()):
        raise ValueError("bounds must be a finite d-by-2 tensor")
    if not bool((bounds[:,1] > bounds[:,0]).all()) or type(points) is not int or points < 2:
        raise ValueError("require increasing bounds and at least two samples")
    splits = []
    for name in ('train','validation','test'):
        key = f'{equation}|{int(seed)}|{name}'.encode()
        stream_seed = int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % (2**63-1)
        generator = torch.Generator(device='cpu').manual_seed(stream_seed)
        x = bounds[:,0] + torch.rand(points,len(bounds),generator=generator,dtype=dtype)*(bounds[:,1]-bounds[:,0])
        y = function(x)
        if y.shape == (points,):y = y.unsqueeze(-1)
        splits.append(Split(x,y))
    return RegressionData(*splits,equation,int(seed))


def regression_metrics(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    variance = target.var(unbiased=False)
    if not bool(torch.isfinite(variance)) or float(variance) <= 0:
        raise ValueError("NMSE requires positive finite target variance")
    mse = (prediction-target).square().mean()
    return dict(mse=float(mse.detach()),nmse=float((mse/variance).detach()))


def geometric_mean_nmse(values, floor=1.0e-10):
    values = torch.as_tensor(values,dtype=torch.float64)
    if not values.numel() or not bool(torch.isfinite(values).all() & (values >= 0).all()):
        raise ValueError("GNMSE requires nonempty finite nonnegative scores")
    return float(values.clamp_min(floor).log().mean().exp())
