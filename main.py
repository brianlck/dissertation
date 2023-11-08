from mcmd.sampler import CMCD
from mcmd.score import ResNet
from mcmd.densities import GaussianMixture
from mcmd.anneal import GeometricAnnealing

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt


score_fn = ResNet(2, 38, 40, 8)
score_fn.reset_parameters()

initial_dist = MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]),
    covariance_matrix=(torch.eye(2) * 2 * 2),
    validate_args=False,
)

target = GaussianMixture()

anneal = GeometricAnnealing()

sampler = CMCD(initial_dist, target, anneal, score_fn, 8, 0.05)

optim = torch.optim.Adam(sampler.parameters(), lr=1e-3)

for i in range(10000):
    optim.zero_grad()
    samples = sampler.sample(300)
    loss = samples.loss
    assert loss != None
    loss.backward()
    optim.step()

    print(loss.item(), sampler.score_fn.scale.item())
