from abc import ABC, abstractmethod
import torch

Tensor = torch.Tensor


class AnnealingSchedule(ABC):
    @abstractmethod
    def interpolate_log_density(
        self, initial: Tensor, target: Tensor, time: float
    ) -> Tensor:
        raise NotImplementedError

    def __call__(self, initial, target, timestep: int) -> Tensor:
        return self.interpolate_log_density(initial, target, timestep)


class GeometricAnnealing(AnnealingSchedule):
    def interpolate_log_density(
        self, initial: Tensor, target: Tensor, t: float
    ) -> Tensor:
        return (1.0 - t) * initial + t * target


class NoAnnealing(AnnealingSchedule):
    def interpolate_log_density(
        self, initial: Tensor, target: Tensor, t: float
    ) -> Tensor:
        return target
