import torch
import torch.nn as nn

from cmcd.densities import log_normal


class GaussianVI(nn.Module):
    def __init__(self, dim, scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale = nn.Parameter(torch.ones(dim) * torch.tensor(scale).log())
        self.mean = nn.Parameter(torch.zeros(dim))

        _mean = torch.tensor(0.0)
        _scale = torch.tensor(1.0)

        self.register_buffer("_scale", _scale)
        self.register_buffer("_mean", _mean)

        self.norm = torch.distributions.Normal(loc=self._mean, scale=self._scale)
        self.dim = dim

    def sample(self, batch):
        return (
            self.norm.sample(torch.Size((batch, self.dim))) * self.scale.exp()
            + self.mean
        )

    def log_prob(self, x):
        return log_normal(x, self.mean, self.scale.exp()).sum(-1)


def prepare_init_dist(config):
    target = config['target']
    if target == "gmm":
        return GaussianVI(2, 2.0)
    elif target == "40gmm":
        return GaussianVI(2, 15.0)
    elif target == "funnel":
        return GaussianVI(10, 1.0)
    elif target == "dw":
        return GaussianVI(config["dw_d"], 3.0)
    else: 
        raise NotImplementedError



