from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.distributions import Normal, MultivariateNormal
import einops

Tensor = torch.Tensor


class LogDensity(ABC):
    def __call__(self, x: torch.Tensor):
        return self.log_density(x)

    @abstractmethod
    def log_density(self, x: torch.Tensor):
        raise NotImplementedError


class Funnel(LogDensity):
    """The funnel distribution from https://arxiv.org/abs/physics/0009028.

    num_dim should be 10. config is unused in this case.
    """

    σ_f = 3
    μ = 0.0
    d = 10

    x1_dist = Normal(loc=μ, scale=σ_f, validate_args=False)

    def __init__(self) -> None:
        super().__init__()

    def log_density(self, x: Tensor):  # x: [B x 10]
        # log density of x1
        log_p1 = Funnel.x1_dist.log_prob(x[:, 0])

        # log density of x[2-10] | x1

        log_p2 = MultivariateNormal(
            loc=torch.zeros(Funnel.d - 1),
            covariance_matrix=x[:, 0].exp().view(-1, 1, 1)
            * torch.eye(Funnel.d - 1).repeat,
        ).log_prob(x[:, 1:])

        return log_p1 + log_p2


class GaussianMixture(LogDensity):
    dist1 = MultivariateNormal(
        loc=torch.tensor([3.0, 0.0]),
        covariance_matrix=torch.tensor([[0.7, 0.0], [0.0, 0.05]]),
        validate_args=False,
    )

    dist2 = MultivariateNormal(
        loc=torch.tensor([-2.5, 0.0]),
        covariance_matrix=torch.tensor([[0.7, 0.0], [0.0, 0.05]]),
        validate_args=False,
    )

    dist3 = MultivariateNormal(
        loc=torch.tensor([2.0, 3.0]),
        covariance_matrix=torch.tensor([[1.0, 0.95], [0.95, 1.0]]),
        validate_args=False,
    )

    def __init__(self) -> None:
        super().__init__()

    def asymmetric_density(self, x):
        logp1 = GaussianMixture.dist1.log_prob(x) - np.log(6.0)
        logp2 = GaussianMixture.dist2.log_prob(x) - np.log(6.0)
        logp3 = GaussianMixture.dist3.log_prob(x) - np.log(6.0)

        return torch.stack([logp1, logp2, logp3])

    def log_density(self, x):
        logp1 = self.asymmetric_density(x) - np.log(2.0)
        logp2 = self.asymmetric_density(torch.flip(x, dims=[-1])) - np.log(2.0)

        return torch.concatenate([logp1, logp2]).logsumexp(dim=0)
