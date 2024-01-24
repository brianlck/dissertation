import torch
from cmcd.samples import Samples
from cmcd.loss_fn import SubtrajectoryBalance


class FakeSampler:

    def __init__(self) -> None:
        self.ln_z=0.3
        self.betas= lambda: torch.tensor([0.0, 0.5, 1.0])


if __name__ == '__main__':
    import os
    os.chdir('../')

    samples = Samples(
        ln_rnd=torch.empty((0,)),
        ln_pi=torch.tensor([[0.3, 0.4, 0.6], [0.1, 0.9, 0.4]]).T,
        ln_ratio=torch.tensor([[0.3, 0.4], [0.9, 0.4]]).T,
        trajectory=torch.empty((0,)),
        ln_forward=torch.empty((0,))
    )    

    sampler = FakeSampler()
    print(SubtrajectoryBalance(2).evaluate(sampler, samples)) # type: ignore
