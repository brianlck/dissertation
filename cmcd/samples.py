from dataclasses import dataclass
import torch    
from torch import Tensor
from numericals import effective_sample_size

@dataclass
class Samples:
    """Return value of Sampler.samples.
    A loss tensor is included as the loss is typically computed during the sampling process
    """

    ln_rnd: Tensor  # [batch_size]
    trajectory: list[Tensor]  # [Time, batch_size, dimension]
    ln_ratio: list[Tensor]
    ln_pi: list[Tensor]

    @property
    def elbo(self):
        return self.ln_rnd.mean()

    @property
    def ln_z(self):
        return torch.logsumexp(self.ln_rnd, dim=-1) - torch.log(
            torch.tensor(self.ln_rnd.shape[-1])
        )

    @property
    def particles(self):
        return self.trajectory[-1]

    def jensen(self, log_norm):
        ln_rnd = self.ln_rnd + log_norm
        device = self.ln_rnd.device
        B = ln_rnd.shape[0]
        ln_t1 = torch.vstack([ln_rnd, torch.zeros(B)]).logsumexp(dim=0)
        log_f = torch.tensor(2.0).log() - ln_t1
        return 0.5 * (ln_rnd.exp() * (log_f + ln_rnd) + log_f).mean()

    def ess(self):
        return effective_sample_size(self.ln_rnd)
