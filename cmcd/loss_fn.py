import abc
import torch
from mcmd.samples import Samples

def _detach_trajectory(samples: Samples):
    samples.trajectory = [x.detach() for x in samples.trajectory]

class LossFunction(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, samples: Samples) -> torch.Tensor:
        raise NotImplementedError

class ReverseKL(LossFunction):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self, samples: Samples):
        return (-samples.ln_rnd).mean()
    
class LogVarianceLoss(LossFunction):
    def __init__(self, detach=True):
        super().__init__()
        self.detach = detach
    
    def evaluate(self, samples: Samples):
        if self.detach:
            _detach_trajectory(samples)
        
        return samples.ln_rnd.var()

class SubtrajLogVar(LossFunction):
    def __init__(self, detach=True, discount=1.06) -> None:
        super().__init__()
        self.detach = detach
        self.discount = discount

    def evaluate(self, samples: Samples):
        if self.detach:
            _detach_trajectory(samples)

        ln_ratio = torch.vstack(samples.ln_ratio).cumsum(dim=0)
        ln_pi = torch.vstack(samples.ln_pi)        

        loss = 0.0
        norm = 0.0

        for i in range(ln_ratio.shape[0]):
            for j in range(i, ln_ratio.shape[0]):
                w = ln_ratio[j] - (ln_ratio[i-1] if i > 0 else 0)
                w += ln_pi[j+1] - ln_pi[i]
                factor = self.discount ** (j - i + 1)
                norm += factor
                loss += factor * (-w).var()
        
        return loss

def prepare_loss_fn(config: dict):
    loss_fn = config['loss_fn']
    if loss_fn == "reverse-kl":
        return ReverseKL()
    elif loss_fn == "log-var":
        return LogVarianceLoss()
    elif loss_fn == "subtraj-log-var":
        return SubtrajLogVar()
    else:
        raise NotImplementedError
