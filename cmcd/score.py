from abc import ABC
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

Tensor = torch.Tensor

class ScoreNetwork(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, t: int) -> Tensor:
        raise NotImplementedError

class PositionalEncoding(nn.Module):
    def __init__(self, max_time_steps: int, embedding_size: int, n: int = 10000) -> None:
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_time_steps).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(max_time_steps, embedding_size, requires_grad=False)
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

        self.pos_embeddings = self.pos_embeddings.cuda()

        # self.linear = nn.Linear(embedding_size, embedding_size)

    def forward(self, t: Tensor) -> Tensor:
        return self.pos_embeddings[t, :]

class ResNet(ScoreNetwork):
    def __init__(
        self, x_dim: int, t_dim: int, h_dim: int, n_bridges: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_timestep = PositionalEncoding(n_bridges + 1, t_dim)
        self.fc1 = nn.Sequential(nn.Linear(x_dim + t_dim, h_dim), nn.BatchNorm1d(h_dim), nn.Softplus())
        self.fc2 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.BatchNorm1d(h_dim), nn.Softplus())
        self.fc3 = nn.Sequential(nn.Linear(h_dim, h_dim), nn.BatchNorm1d(h_dim), nn.Softplus())
        self.fc4 = nn.Linear(h_dim, x_dim)
        self.scale = nn.Parameter(torch.tensor(0.0))

    def prepare_network_input(self, x: Tensor, t: int):
        # x: [B, x_dim]
        B = x.shape[0]

        # embed timestamp
        time_embedding = einops.repeat(self.embed_timestep(t), 'h -> b h', b=B)

        # concat time embedding
        result, _ = einops.pack([x, time_embedding], "b *")

        return result

    def forward(self, inputs: Tensor, t: int):
        """
        Input dimensions
            inputs: [B, x_dim]
        """

        x = self.prepare_network_input(inputs, t)
        
        x = self.fc1(x)

        tmp = x
        x = self.fc2(x)
        x = tmp + x
        
        tmp = x
        x = self.fc3(x)
        x = tmp + x
        
        x = self.fc4(x)

        return self.scale * x

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.fc1[0].weight)
        torch.nn.init.normal_(self.fc1[0].bias)

        torch.nn.init.kaiming_normal_(self.fc2[0].weight)
        torch.nn.init.normal_(self.fc2[0].bias)

        torch.nn.init.kaiming_normal_(self.fc3[0].weight)
        torch.nn.init.normal_(self.fc3[0].bias)

        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.normal_(self.fc4.bias)

def prepare_score_fn(config):
    score_fn = ResNet(config['x_dim'], config['t_dim'], config['h_dim'], config['n_bridges'])
    return score_fn