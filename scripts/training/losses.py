import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaVAELoss(nn.Module):
    def __init__(
        self, recon_loss_weight: float = 10_000.0, kld_loss_weight: float = 1.0
    ) -> None:
        """Constructor.

        Args:
            recon_loss_weight (float, optional): Reconstruction (MSE) loss weight. Defaults to 10_000.0.
            kld_loss_weight (float, optional): KL. Defaults to 1.0.
        """
        super().__init__()
        self.recon_weight = recon_loss_weight
        self.kld_weight = kld_loss_weight

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Vanilla VAE loss.

        Args:
            y_true (torch.Tensor): Target.
            y_pred (torch.Tensor): Prediction.
            mu (torch.Tensor): mu prediction.
            log_var (torch.Tensor): Logarithm of variance prediction.

        Returns:
            t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple with (total_loss, unscaled reconstruction loss, unscaled KL divergence loss)
        """
        recon_loss = self.recon_weight * F.mse_loss(y_true, y_pred)
        kld_loss = self.kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        total_loss = recon_loss + kld_loss

        return total_loss, recon_loss.detach(), kld_loss.detach()


class VanillaVAELossWithEdges(nn.Module):
    def __init__(
        self, recon_loss_weight: float = 10_000.0, kld_loss_weight: float = 1.0
    ) -> None:
        """Constructor.

        Args:
            recon_loss_weight (float, optional): Reconstruction (MSE) loss weight. Defaults to 10_000.0.
            kld_loss_weight (float, optional): KL. Defaults to 1.0.
        """
        super().__init__()
        self.recon_weight = recon_loss_weight
        self.kld_weight = kld_loss_weight

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        edges_weights: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes Vanilla VAE loss.

        Args:
            y_true (torch.Tensor): Target.
            y_pred (torch.Tensor): Prediction.
            mu (torch.Tensor): mu prediction.
            log_var (torch.Tensor): Logarithm of variance prediction.
            edges_weights (torch.Tensor): Edges from target.

        Returns:
            t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple with (total_loss, unscaled reconstruction loss, unscaled KL divergence loss)
        """
        recon_loss = (
            self.recon_weight
            * ((y_true - y_pred) ** 2 * edges_weights.unsqueeze(dim=1)).mean()
        )

        kld_loss = self.kld_weight * torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        total_loss = recon_loss + kld_loss

        return total_loss, recon_loss.detach(), kld_loss.detach()
