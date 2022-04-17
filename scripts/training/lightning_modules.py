import typing as t

import pytorch_lightning as pl
import torch
import torchmetrics

import face_autoencoder.model as mdl
from scripts.training import losses


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        batch_size: int,
        num_workers: int,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: t.Optional[str] = None) -> None:
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        *,
        lr: float,
        nn_input_image_resolution: int,
        latent_dim: int,
        hidden_dims: t.List[int] = [32, 64, 128, 256, 512],
        recon_loss_weight: float = 10_000.0,
        kld_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = lr
        self.criterion = losses.VanillaVAELoss(recon_loss_weight, kld_loss_weight)

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        self.neural_net = mdl.VanillaVAE(
            nn_input_image_resolution=nn_input_image_resolution,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.neural_net.parameters(), lr=self.learning_rate)

    def training_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch)

    def validation_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch)

    def test_step(
        self, batch: t.Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> pl.utilities.types.STEP_OUTPUT:
        return self._step(batch)

    def training_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="train", outputs=outputs, mse_metric=self.train_mse
        )

    def validation_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="val", outputs=outputs, mse_metric=self.val_mse
        )
        pass

    def test_epoch_end(self, outputs: pl.utilities.types.EPOCH_OUTPUT) -> None:
        self._summarize_epoch(
            log_prefix="test", outputs=outputs, mse_metric=self.test_mse
        )

    def _step(
        self,
        batch: t.Tuple[torch.Tensor, torch.Tensor],
        mse_metric: torchmetrics.MeanSquaredError,
    ) -> pl.utilities.types.STEP_OUTPUT:
        image, px_weights = batch

        reconstruction = self.neural_net(image)

        loss = self.criterion(reconstruction, image, px_weights)

        mse_metric(reconstruction, image)

        return {"loss": loss}

    def _summarize_epoch(
        self,
        log_prefix: str,
        outputs: pl.utilities.types.EPOCH_OUTPUT,
        mse_metric: torchmetrics.MeanSquaredError,
    ) -> None:
        mean_loss = torch.tensor([out["loss"] for out in outputs]).mean()
        self.log(f"{log_prefix}_loss", mean_loss, on_epoch=True)

        mse = mse_metric.compute()

        self.log(f"{log_prefix}_mse", mse, on_epoch=True)

        mse_metric.reset()
