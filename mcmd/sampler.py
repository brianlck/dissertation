from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass
import einops
import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
from mcmd.anneal import AnnealingSchedule
from mcmd.densities import LogDensity
from mcmd.score import ScoreNetwork
import torch.autograd.profiler as profiler

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution


@dataclass
class Samples:
    """Return value of Sampler.samples.
    A loss tensor is included as the loss is typically computed during the sampling process
    """

    particles: Tensor  # [batch_size, particle_size]
    weights: Tensor  # [batch_size]
    loss: Optional[Tensor]  # [batch_size]; None in inference or for non-learnt sampler


class Sampler(ABC):
    def __init__(
        self, initial_dist: Distribution, target_density: LogDensity, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.initial_dist = initial_dist
        self.target_density = target_density

    @abstractmethod
    def evolve(self, particles: Tensor) -> Samples:
        pass

    def sample(self, batch: int):
        particles = self.initial_dist.sample_n(batch)
        return self.evolve(particles)


class CMCD(Sampler, nn.Module):
    def __init__(
        self,
        initial_dist: Distribution,
        target_density: LogDensity,
        anneal_schedule: AnnealingSchedule,
        score_network: ScoreNetwork,
        n_bridges: int,
        initial_eps: float,
        *args,
        **kwargs
    ) -> None:
        super().__init__(initial_dist, target_density, *args, **kwargs)
        self.score_fn = score_network
        self.anneal = anneal_schedule
        self.n_bridges = n_bridges

        # TODO: provide better schedules
        self.eps = nn.Parameter(torch.tensor(initial_eps).log())

        self.grid_t = nn.Parameter(torch.ones(n_bridges))

    def betas(self):
        zero = torch.tensor([0.0])
        betas = torch.concat([zero, self.grid_t])
        betas = torch.cumsum(betas, dim=0) / torch.sum(self.grid_t)
        return betas

    def drift(self, particles, i):
        return (
            self.grad_log_phi(particles, self.betas()[i]) - self.score_fn(particles, i)
        ) * torch.exp(self.eps)

    noise_dist = Normal(
        loc=torch.tensor(0.0),
        scale=torch.tensor(1.0),
        validate_args=False,
    )

    def prepare_noise(self, N, B):
        size = torch.Size((self.n_bridges, B, N))
        return CMCD.noise_dist.sample(size)

    def log_phi(self, x, t):
        log_initial = self.initial_dist.log_prob(x)
        log_target = self.target_density.log_density(x)

        return self.anneal(log_initial, log_target, t)

    # compute the gradient of log_phi wrt x
    grad_log_phi = torch.func.vmap(torch.func.grad(log_phi, argnums=1), in_dims=(None, 0, None))  # type: ignore

    def evolve(self, particles: Tensor):
        B = particles.shape[0]
        N = particles.shape[1]

        w = -self.initial_dist.log_prob(particles)
        dt = torch.exp(self.eps)

        betas = self.betas()
        noises = self.prepare_noise(N, B)

        for i in range(self.n_bridges):
            drift = self.grad_log_phi(particles, betas[i]) - self.score_fn(particles, i)

            forward_kernel = Normal(
                loc=particles + dt * drift,
                scale=torch.sqrt(2 * dt),
                validate_args=False,
            )

            new_particles = particles + dt * drift + torch.sqrt(2 * dt) * noises[i]

            rev_drift = self.grad_log_phi(
                new_particles,
                betas[i + 1],
            ) + self.score_fn(new_particles, i + 1)

            backward_kernel = Normal(
                loc=new_particles + dt * rev_drift,
                scale=torch.sqrt(2 * dt),
                validate_args=False,
            )

            forward_log_prob = einops.reduce(
                forward_kernel.log_prob(new_particles), "b n -> b", "sum"
            )
            backward_log_prob = einops.reduce(
                backward_kernel.log_prob(particles), "b n -> b", "sum"
            )

            w += backward_log_prob - forward_log_prob

            particles = new_particles

        w += self.target_density.log_density(particles)

        return Samples(particles=particles, weights=w, loss=-w.mean())
