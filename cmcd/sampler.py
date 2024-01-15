from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Normal
from mcmd.anneal import AnnealingSchedule
from mcmd.densities import  log_normal
from mcmd.samples import Samples
from mcmd.score import ScoreNetwork

Tensor = torch.Tensor

class Sampler(nn.Module, ABC):
    def __init__(
        self, initial_dist, target_density, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.initial_dist = initial_dist
        self.target_density = target_density

    @abstractmethod
    def evolve(self, particles: Tensor) -> Samples:
        pass

    def sample(self, batch, **kwargs):
        particles = self.initial_dist.sample(batch)
        return self.evolve(particles, **kwargs)


class CMCD(Sampler, nn.Module):
    def __init__(
        self,
        initial_dist,
        target_density,
        anneal_schedule: AnnealingSchedule,
        score_network: ScoreNetwork,
        n_bridges: int,
        initial_eps: float,
        eps_trainable: bool = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(initial_dist, target_density, *args, **kwargs)
        self.score_fn = score_network
        self.anneal = anneal_schedule
        self.n_bridges = n_bridges

        # TODO: provide better schedules
        if eps_trainable:
            self.eps = nn.Parameter(torch.tensor(initial_eps))
        else:
            eps = torch.tensor(initial_eps)
            self.register_buffer("eps", eps)

        self.grid_t = nn.Parameter(torch.zeros(n_bridges))

        noise_loc = torch.tensor(0.0)
        noise_scale = torch.tensor(1.0)

        self.noise_dist = Normal(
            loc=noise_loc,
            scale=noise_scale,
            validate_args=False,
        )

    def repel(self, x, t, a):
        batch = int(x.shape[0] * a)
        repel_x = x[:batch]
        distance_m = torch.cdist(repel_x, repel_x)
        h_t = distance_m.flatten().median() ** 2 / torch.tensor(batch).log()
        d = (-(distance_m**2 / h_t)).exp()
        f = repel_x[:, None] - repel_x[None, :]
        force = -(d[:, :, None] * f).sum(1) / h_t

        return torch.nn.functional.pad(
            force, (0, 0, 0, x.shape[0] - batch), "constant", 0
        )

    def betas(self):
        zero = torch.zeros(1, device=self.grid_t.device)
        sig = torch.nn.functional.sigmoid(self.grid_t)
        betas = torch.concat([zero, sig])
        betas = torch.cumsum(betas, dim=0) / torch.sum(sig)
        return betas

    def drift(self, particles, i, stable=False):
        return (
            self.grad_log_phi_(particles, self.betas()[i], stable)
            - self.score_fn(particles, i)
        ) * self.eps

    def prepare_noise(self, N, B):
        size = torch.Size((self.n_bridges, B, N))
        return self.noise_dist.sample(size)

    def log_phi(self, x, t):
        log_initial = self.initial_dist.log_prob(x)
        log_target = self.target_density.log_density(x)

        return self.anneal(log_initial, log_target, t)

    def get_grad(f):
        @torch._dynamo.allow_in_graph
        def _inner(self, x, t):
            return torch.func.grad(lambda x, t: f(self, x, t).sum())(x, t) # type: ignore

        return lambda self, x, t: _inner(self, x, t)

    def get_grad2(f):
        @torch._dynamo.allow_in_graph
        def _inner(self, x):
            return torch.func.grad(lambda x: f(self, x).sum())(x) # type: ignore

        return lambda self, x: _inner(self, x)

    def log_init_density(self, x):
        return self.initial_dist.log_prob(x)

    def log_target_density(self, x):
        return self.target_density.log_density(x)

    grad_log_phi = get_grad(log_phi)  # type: ignore
    grad_target_log = get_grad2(log_target_density) # type: ignore
    grad_initial_log = get_grad2(log_init_density) # type: ignore

    def correct(self, x, t):
        dt = self.eps
        for i in range(5):
            noise = self.noise_dist.sample(torch.Size((x.shape[0], x.shape[1])))
            x = x + self.grad_log_phi(x, t) * dt + torch.sqrt(2 * dt) * noise
        return x

    def clipped_grad(self, x, t):
        grad_log_init: Tensor = self.grad_initial_log(x)
        grad_log_target: Tensor = self.grad_target_log(x)

        tmp = torch.clip(grad_log_target.norm(dim=1)[:, None], min=0, max=1e2)
        grad_log_target = torch.nn.functional.normalize(grad_log_target) * torch.clip(
            tmp, min=0, max=1e2
        )

        tmp = torch.clip(grad_log_init.norm(dim=1)[:, None], min=0, max=1e2)
        grad_log_init = torch.nn.functional.normalize(grad_log_init) * torch.clip(
            tmp, min=0, max=1e2
        )

        return (1 - t) * grad_log_init + t * grad_log_target

    def grad_log_phi_(self, x, t, stable):
        if stable:
            return self.clipped_grad(x, t)
        else:
            return self.grad_log_phi(x, t)

    def _clamp_vec(self, v):
        tmp = torch.clip(v.norm(dim=1)[:, None], min=0, max=1e2)
        return torch.nn.functional.normalize(v) * torch.clip(tmp, min=0, max=1e2)

    def step(
        self,
        particles,
        betas,
        noises,
        dt,
        i,
        repel,
        repel_percentage,
        precomputed_grad_phi=None,
        stable=False,
    ):
        current_grad_phi = (
            self.grad_log_phi_(particles, betas[i], stable)
            if precomputed_grad_phi is None
            else precomputed_grad_phi
        )
        drift = current_grad_phi - self.score_fn(particles, i)
        new_particles = particles + dt * drift + torch.sqrt(2 * dt) * noises[i]

        new_particles = (
            new_particles
            - self.repel(particles, i / self.n_bridges, repel_percentage) * dt
            if repel
            else new_particles
        )

        next_grad_phi = self.grad_log_phi_(new_particles, betas[i + 1], stable)
        rev_drift = next_grad_phi + self.score_fn(new_particles, i + 1)

        forward_log_prob = log_normal(
            new_particles, particles + dt * drift, torch.sqrt(2 * dt)
        ).sum(-1)
        backward_log_prob = log_normal(
            particles, new_particles + dt * rev_drift, torch.sqrt(2 * dt)
        ).sum(-1)

        ln_ratio = backward_log_prob - forward_log_prob

        return ln_ratio, new_particles, current_grad_phi, next_grad_phi

    def evolve(self, particles: Tensor, repel: bool, repel_percentage: float):
        B = particles.shape[0]
        N = particles.shape[1]

        w = -self.initial_dist.log_prob(particles)
        dt = self.eps

        betas = self.betas()
        noises = self.prepare_noise(N, B)

        trajectory = [particles]
        ln_ratio = []
        ln_pi = []

        next_grad_phi = None
        for i in range(self.n_bridges):
            current_ln_ratio, particles, current_grad_phi, next_grad_phi = self.step(
                particles,
                betas,
                noises,
                dt,
                i,
                repel,
                repel_percentage,
                next_grad_phi,
            )

            ln_ratio.append(current_ln_ratio)
            ln_pi.append(current_grad_phi)
            trajectory.append(particles)
            w += current_ln_ratio

        target_log_density = self.target_density.log_density(particles)
        ln_pi.append(target_log_density)
        w += target_log_density

        result = Samples(
            ln_rnd=w, trajectory=trajectory, ln_ratio=ln_ratio, ln_pi=ln_pi
        )

        return result
