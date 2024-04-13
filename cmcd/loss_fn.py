import abc
import torch
import os
from torch._tensor import Tensor
from cmcd.sampler import CMCD, Sampler
from cmcd.samples import Samples
import numpy as np


class LossFunction(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, sampler: Sampler, samples: Samples) -> torch.Tensor:
        raise NotImplementedError

    def calculate_priority(self, sampler: Sampler, samples: Samples) -> torch.Tensor:
        raise NotImplementedError


class ReverseKL(LossFunction):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, sampler: Sampler, samples: Samples):
        return (-samples.ln_rnd).mean()


class LogVarianceLoss(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        
        return samples.ln_rnd.var()
        # return ((samples.ln_rnd - samples.ln_z) ** 2).mean()


class ImprovedLogVarianceLoss(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        return ((samples.ln_rnd - samples.ln_z) ** 2).mean()
class ImprovedTrajectoryBalance(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach

    def evaluate(self, sampler: CMCD, samples: Samples):

        return ((samples.ln_rnd - sampler.ln_z_base) ** 2).mean()


class SubtrajLogVar(LossFunction):
    def __init__(self, detach=True, discount=1.06) -> None:
        super().__init__()
        self.detach = detach
        self.discount = discount

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        ln_ratio = samples.ln_ratio.cumsum(dim=0)
        ln_pi = samples.ln_pi

        loss = 0.0
        norm = 0.0

        for i in range(ln_ratio.shape[0]):
            for j in range(i, ln_ratio.shape[0]):
                w = ln_ratio[j] - (ln_ratio[i - 1] if i > 0 else 0)
                w += ln_pi[j + 1] - ln_pi[i]
                factor = self.discount ** (j - i + 1)
                norm += factor
                loss += factor * (-w).var()

        return loss


class TrajectoryBalance(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        return (samples.ln_rnd - sampler.ln_z[-1]).pow(2).mean()


def cal_subtb_coef_matrix(lamda, N):
    """
    diff_matrix: (N+1, N+1)
    0, 1, 2, ...
    -1, 0, 1, ...
    -2, -1, 0, ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    range_vals = torch.arange(0, N + 1)
    diff_matrix = range_vals - range_vals.view(-1, 1)
    B = np.log(lamda) * diff_matrix
    B[diff_matrix <= 0] = -np.inf
    log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
    coef = torch.exp(B - log_total_lambda)
    return coef


class SubtrajectoryBalance(LossFunction):
    def __init__(self, n_bridge: int, detach=True, discount=2.0):
        super().__init__()
        self.detach = detach
        self.coef = cal_subtb_coef_matrix(discount, n_bridge)

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        ln_ratio = torch.vstack([torch.zeros(1, samples.ln_ratio.shape[1]), -samples.ln_ratio.cumsum(dim=0)])
        ln_pi = samples.ln_pi + sampler.ln_z.view(-1, 1)
        betas = sampler.betas()
        ln_pi = samples.ln_pi + torch.concat([torch.zeros((1,)), sampler.ln_z])
        ln_pi = torch.swapaxes(ln_pi, 0, 1)
        ln_ratio = ln_ratio.swapaxes(0, 1)

        A1 = ln_pi.unsqueeze(1) - ln_pi.unsqueeze(2)  # (b, T+1, T+1)
        A2 = ln_ratio[:, :, None] - ln_ratio[:, None, :] + A1  # (b, T+1, T+1)


        A2 = (A2).pow(2).mean(dim=0)  # (T+1, T+1)

        loss = torch.triu(A2 * self.coef, diagonal=1).sum()

        return loss

class ScoreMatching(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach

    def evaluate(self, sampler: Sampler, samples: Samples):
        if self.detach:
            samples.trajectory = samples.trajectory.detach()

        # print(samples.trajectory)
        # print(samples.score.pow(2))
        return samples.score.pow(2).sum(dim=-1).mean(dim=-1).mean()

def prepare_loss_fn(config: dict):
    loss_fn = config["loss_fn"]
    if loss_fn == "reverse-kl":
        return ReverseKL()
    elif loss_fn == "score-matching":
        return ScoreMatching()
    elif loss_fn == "log-var":
        return LogVarianceLoss()
    elif loss_fn == "improved-log-var":
        return ImprovedLogVarianceLoss()
    elif loss_fn == "subtraj-log-var":
        return SubtrajLogVar()
    elif loss_fn == "traj-balance":
        return TrajectoryBalance()
    elif loss_fn == "improved-traj-balance":
        return ImprovedTrajectoryBalance()
    elif loss_fn == "subtraj-balance":
        return SubtrajectoryBalance(config["n_bridges"])
    else:
        raise NotImplementedError

if __name__ == '__main__':
    import os
    os.chdir('../')

    samples = Samples(
        ln_rnd=torch.empty((0,)),
        ln_pi=torch.tensor([[0.3, 0.4, 0.6], [0.1, 0.9, 0.4]]).T,
        ln_ratio=torch.tensor([[0.3, 0.4], [0.9, 0.4]]).T,
        trajectory=torch.empty((0,)),
        ln_forward=torch.empty((0,)),
        score=None
    )    

    sampler = dict(
        ln_z=0.3
    )

    print(SubtrajectoryBalance(2).evaluate(sampler, samples)) # type: ignore
