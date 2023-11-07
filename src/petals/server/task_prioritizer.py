from abc import ABC, abstractmethod

import torch


class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose responsibility is to evaluate task priority"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amount of points given, task input and additional kwargs. Lower priority is better"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        # Inference steps go first since they are more latency-sensitive
        return 1.0 if kwargs.get("type") == "inference" else 2.0
