from abc import ABC
from unittest import result
import torch
import torch.nn as nn
import einops

Tensor = torch.Tensor


class ScoreNetwork(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, t: int) -> Tensor:
        raise NotImplementedError


# class Residual(nn.Module):


class ResNet(ScoreNetwork):
    def __init__(
        self, x_dim: int, t_dim: int, h_dim: int, n_bridges: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_timestep = nn.Embedding(n_bridges + 1, t_dim)
        self.fc1 = nn.Sequential(nn.Linear(x_dim + t_dim, h_dim), nn.Softplus())
        self.norm1 = nn.LayerNorm(h_dim)
        self.fc2 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Softplus())
        self.norm2 = nn.LayerNorm(h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def prepare_network_input(self, x: Tensor, t: int):
        # x: [B, x_dim]
        B = x.shape[0]

        # embed timestamp
        time_embedding = self.embed_timestep(torch.full([B], t))

        # concat time embedding
        result, _ = einops.pack([x, time_embedding], "b *")

        return result

    def forward(self, inputs: Tensor, t: int):
        """
        Input dimensions
            inputs: [B, x_dim]
        """

        x = self.prepare_network_input(inputs, t)
        x = self.norm1(x + self.fc1(x))
        x = self.norm2(x + self.fc2(x))
        x = self.fc3(x)

        return self.scale * x

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.fc1[0].weight)
        torch.nn.init.normal_(self.fc1[0].bias)

        torch.nn.init.xavier_normal_(self.fc2[0].weight)
        torch.nn.init.normal_(self.fc2[0].bias)

        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias)
