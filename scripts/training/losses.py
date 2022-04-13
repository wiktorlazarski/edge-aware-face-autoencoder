import torch
import torch.nn as nn


class VanillaVAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        pass
