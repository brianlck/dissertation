from git import Optional
import torch
import numpy as np
from cmcd.sampler import Sampler

from cmcd.samples import Samples


# Gumbel softmax by some oxford person
def sample_without_replacement(logits: torch.Tensor, n: int) -> torch.Tensor:
    z = torch.distributions.Gumbel(torch.tensor(0.0), torch.tensor(1.0)).sample(logits.shape)
    topk = torch.topk(z + logits, n, sorted=False)
    indices = topk.indices
    indices = indices[torch.randperm(n).to(indices.device)]
    return indices

class ReplayBuffer:
    def __init__(self, traj_len, dim, capacity):
        self.capacity = capacity
        self.prio = torch.zeros((capacity,))
        self.paths = torch.zeros((capacity, traj_len, dim))
        self.size = 0

    def __len__(self):
        return min(self.capacity, self.size)


    @torch.no_grad()
    def calculate_priority(self, sampler: Sampler, samples: Samples):
        return samples.ln_rnd
        return samples.ln_rnd

    @torch.no_grad()
    @torch.compiler.allow_in_graph
    def add(self, sampler, samples, indicies: Optional[torch.Tensor] = None):
        priority, paths = self.calculate_priority(sampler, samples).detach(), samples.trajectory.swapaxes(0, 1).detach()
        if indicies is not None:
            self.prio[indicies] = priority[:indicies.shape[0]]
            priority = priority[indicies.shape[0]:]
            paths = paths[indicies.shape[0]:]
        self.size += priority.shape[0]
        
        insert_position = torch.arange(self.size, self.size + len(priority)) % self.capacity
        self.prio[insert_position] = priority
        self.paths[insert_position] = paths

        
    @torch.no_grad()
    def sample(self, batch_size, uniform=True):
        priority = self.prio[: min(self.capacity, self.size)] 
        selected_id = sample_without_replacement(priority, batch_size)
        paths = self.paths[selected_id].swapaxes(0, 1)
        return selected_id, paths
