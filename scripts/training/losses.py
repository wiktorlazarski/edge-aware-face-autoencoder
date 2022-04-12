import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        pass
