from abc import ABC, abstractmethod
from statistics import covariance
import torch
import torch.nn as nn
import numpy as np
from mcmd.numericals import  quadratic_function, importance_weighted_expectation, effective_sample_size_over_p, MC_estimate_true_expectation

Tensor = torch.Tensor

class LogDensity(ABC):
    def __call__(self, x: torch.Tensor):
        return self.log_density(x)

    @abstractmethod
    def log_density(self, x: torch.Tensor):
        raise NotImplementedError

def log_normal(x, loc, scale):
    dx = x - loc
    melan = -dx * dx / (2.0 * scale * scale)
    norm = -torch.log(scale) - 1/2 * torch.log(torch.tensor(2 * torch.pi, device=x.device))
    return (melan + norm)
    


def log_normal_density(x: Tensor, m: Tensor, cov: Tensor):
    L = torch.linalg.cholesky(cov)
    y = torch.linalg.solve_triangular(L, (x - m).unsqueeze(-1), upper=False).squeeze()
    mahalanobis = -1/2 * torch.einsum('...i, ...i->...', y, y)
    n = m.shape[-1]
    norm = -n/2 * torch.log(torch.tensor(2 * torch.pi)) - torch.log(L.diagonal(dim1=-2, dim2=-1)).sum(dim=-1)
    return mahalanobis + norm 


class Funnel(LogDensity):
    """ The funnel distribution from https://arxiv.org/abs/physics/0009028."""
    def __init__(self) -> None:
        super().__init__()

    def log_density(self, x: Tensor):  # x: [B x 10]
        def const(f):
            return torch.tensor(f, device=x.device)
        log_p1 = log_normal(x[:, 0], const(0.0), const(3.0))

        log_p2 = log_normal(x[:, 1:], const(0.0), x[:, 0].exp().unsqueeze(-1).sqrt()).sum(dim=-1) 

        log_density = log_p1 + log_p2

        assert log_p1.shape == log_p2.shape
        return log_density

    def norm_constant(self):
        batch = 5
        dim = 2
        truth = 0.0
        dist1 = torch.distributions.Normal(loc=0.0, scale=3.0) 
        x1 = dist1.sample(torch.Size((batch, )))
        dist2 = torch.distributions.MultivariateNormal(loc=torch.zeros(dim-1), covariance_matrix=torch.eye(dim-1) * x1.exp().reshape(-1, 1, 1))
        x_rest = dist2.sample()
        truth = dist1.log_prob(x1) + dist2.log_prob(x_rest)
        x = torch.concat([x1.unsqueeze(-1), x_rest], dim=1)
        log_density = self.log_density(x)
        print('x1', dist1.log_prob(x1))
        print('x_rest', dist2.log_prob(x_rest))
        print('truth', truth)
        print('log_density', log_density)
        print(((truth - log_density) ** 2).mean())
        integral = log_density.exp().mean()
        # for i in range(dim):
        #     integral = integral * 200

        return integral

    def log_norm(self):
        return 0.0



class GaussianMixture(LogDensity):
    def __init__(self) -> None:
        super().__init__()
        self.m1 = torch.tensor([3.0, 0.0])
        self.m2 = torch.tensor([-2.5, 0.0])
        self.m3 = torch.tensor([2.0, 3.0])

        self.cov1 = torch.tensor([[0.7, 0.0], [0.0, 0.05]])
        self.cov2 = torch.tensor([[0.7, 0.0], [0.0, 0.05]])
        self.cov3 = torch.tensor([[1.0, 0.95], [0.95, 1.0]])

    
    def asymmetric_density(self, x):
        norm = torch.log(torch.tensor(6.0, device=x.device))
        logp1 = log_normal_density(x, self.m1, self.cov1) - norm
        logp2 = log_normal_density(x, self.m2, self.cov2) - norm
        logp3 = log_normal_density(x, self.m3, self.cov3) - norm

        return torch.stack([logp1, logp2, logp3])
    
    def log_density(self, x):
        logp1 = self.asymmetric_density(x)
        logp2 = self.asymmetric_density(torch.flip(x, dims=[-1]))

        return torch.concatenate([logp1, logp2]).logsumexp(dim=0)

    def log_norm(self, x):
        return 0.


class DoubleWell(LogDensity):
    def __init__(self, d, m, delta):
        self.d = d
        self.m = m
        self.delta = torch.tensor(delta)

    
    def log_density(self, x):
        m = self.m
        d = self.d
        delta = self.delta

        prefix = x[:, :m]

        k = ((prefix ** 2 - delta) ** 2).sum(1)
        suffix = x[:, m:]
        k2 = 0.5 * (suffix * suffix).sum(1)

        return -k - k2

    def log_norm(self):
        l, r = -100, 100
        s = 100000000
        pt = torch.distributions.uniform.Uniform(l, r).sample(torch.Size((s, )))
        fst = ((-(pt * pt - self.delta) ** 2).exp() * ((r - l) / s)).sum().log()
        
        pt = torch.distributions.uniform.Uniform(l, r).sample(torch.Size((s, )))
        snd = ((-0.5 * pt * pt).exp() * ((r - l) / s)).sum().log()

        return fst * self.m + snd * (self.d - self.m)
    

class GMM(nn.Module, LogDensity):
    def __init__(self, dim, n_mixes, loc_scaling, log_var_scaling=0.1, seed=0,
                 n_test_set_samples=1000, use_gpu=True,
                 true_expectation_estimation_n_samples=int(1e7)):
        super(GMM, self).__init__()
        self.seed = seed
        self.n_mixes = n_mixes
        self.dim = dim
        self.n_test_set_samples = n_test_set_samples

        mean = (torch.rand((n_mixes, dim)) - 0.5)*2 * loc_scaling
        log_var = torch.ones((n_mixes, dim)) * log_var_scaling

        self.register_buffer("cat_probs", torch.ones(n_mixes))
        self.register_buffer("locs", mean)
        self.register_buffer("scale_trils", torch.diag_embed(torch.nn.functional.softplus(log_var)))
        self.expectation_function = quadratic_function
        self.register_buffer("true_expectation", MC_estimate_true_expectation(self,
                                                             self.expectation_function,
                                                             true_expectation_estimation_n_samples
                                                                              ))
        self.device = "cuda" if use_gpu else "cpu"
        self.to(self.device)

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def distribution(self):
        mix = torch.distributions.Categorical(self.cat_probs)
        com = torch.distributions.MultivariateNormal(self.locs,
                                                     scale_tril=self.scale_trils,
                                                     validate_args=False)
        return torch.distributions.MixtureSameFamily(mixture_distribution=mix,
                                                     component_distribution=com,
                                                     validate_args=False)

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples, ))

    def log_density(self, x: torch.Tensor):
        log_prob = self.distribution.log_prob(x)
        # Very low probability samples can cause issues (we turn off validate_args of the
        # distribution object which typically raises an expection related to this.
        # We manually decrease the distributions log prob to prevent them having an effect on
        # the loss/buffer.
        mask = torch.zeros_like(log_prob)
        mask[log_prob < -1e4] = - torch.tensor(float("inf"))
        log_prob = log_prob + mask
        return log_prob

    def sample(self, shape=(1,)):
        return self.distribution.sample(shape)

    def evaluate_expectation(self, samples, log_w):
        expectation = importance_weighted_expectation(self.expectation_function,
                                                         samples, log_w)
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = (expectation - true_expectation) / true_expectation
        return bias_normed

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn = None,
                            batch_size = None):
        bias_normed = self.evaluate_expectation(samples, log_w)
        bias_no_correction = self.evaluate_expectation(samples, torch.ones_like(log_w))
        if log_q_fn:
            log_q_test = log_q_fn(self.test_set)
            log_p_test = self.log_prob(self.test_set)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "bias_normed": torch.abs(bias_normed).cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
                "ess_over_p": ess_over_p.detach().cpu().item(),
                "kl_forward": kl_forward.detach().cpu().item()
                            }
        else:
            summary_dict = {"bias_normed": bias_normed.cpu().item(),
                            "bias_no_correction": torch.abs(bias_no_correction).cpu().item()}
        return summary_dict
    

def prepare_target(config):
    target = config['target']
    if target == "gmm":
        return GaussianMixture()
    elif target == "funnel":
        return Funnel()
    elif target == "dw":
        return DoubleWell(config["dw_d"], config["dw_m"], config["dw_delta"])
    elif target == "40gmm":
        dim = 2
        n_mixes = 40
        loc_scaling = 40.0  # scale of the problem (changes how far apart the modes of each Guassian component will be)
        log_var_scaling = 1.0  # variance of each Gaussian
        return GMM(
            dim=dim,
            n_mixes=n_mixes,
            loc_scaling=loc_scaling,
            log_var_scaling=log_var_scaling,
            use_gpu=True,
            true_expectation_estimation_n_samples=int(1e5),
        )
    else:
        raise NotImplementedError


def uniform_norm(loc, scale):
    l, r = -100, 100
    points = torch.distributions.uniform.Uniform(l, r).sample(torch.Size((1000, )))
    print(points)
    log_d = log_normal(points, torch.tensor(loc), torch.tensor(scale))
    cum = (r - l) * log_d.exp().mean()
    return cum


