import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEncoderError(Exception):
    """VAEEncoder error."""

    pass


class VAEEncoder(nn.Module):
    class ConvolutionalLayer(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            """Constructor.

            Args:
                in_channels (int): Conv layer input channels.
                out_channels (int): Conv layer output channels.
            """
            super().__init__()

            self.conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            """Forward pass through convolutional layer.

            Args:
                batch (torch.Tensor): Input batch.

            Returns:
                torch.Tensor: Computed feature maps.
            """
            out = self.conv_layer(batch)
            out = self.batch_norm_layer(out)

            return F.leaky_relu(out)

    def __init__(
        self,
        nn_input_image_resolution: int,
        latent_dim: int,
        hidden_dims: t.List[int] = [32, 64, 128, 256, 512],
    ) -> None:
        """Constructor.

        Args:
            nn_input_image_resolution (int): Neural Network input image resolution. Assumes square image.
            latent_dim (int): Latend dimension size.
            hidden_dims (t.List[int], optional): Feature maps' channels hidden dimensions. Defaults to [32, 64, 128, 256, 512].

        Raises:
            VAEEncoderError: Raised if hidden_dims lead to improper final feature map resolution.
        """
        feature_map_res = nn_input_image_resolution // 2 ** len(hidden_dims)
        if not feature_map_res >= 1:
            raise VAEEncoderError(
                "Input image resolution must be divisible by 2 ** len(hidden_dims)"
            )
        super().__init__()

        layers = []
        in_channels = 3
        for hidden_dim in hidden_dims:
            convolutional_layer = self.ConvolutionalLayer(
                in_channels=in_channels, out_channels=hidden_dim
            )

            layers.append(convolutional_layer)

            in_channels = hidden_dim

        self.cnn = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1] * feature_map_res ** 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * feature_map_res ** 2, latent_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Maps batch samples to latent dimension.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Images mapped to latent dimension.
        """
        feature_maps = self.cnn(batch)

        flatten_features = torch.flatten(feature_maps, start_dim=1)

        mu = self.fc_mu(flatten_features)
        log_var = self.fc_var(flatten_features)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        z = eps * std + mu

        return z


class VAEDecoder(nn.Module):
    class TransposeConvolution(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            """Transpose convolutional layer of decoder.

            Args:
                in_channels (int): Number of input channels to transpose conv layer.
                out_channels (int): Number of output channels to transpose conv layer.
            """
            super().__init__()
            self.transpose_conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            """Forwards batch through transpose convolutional layer.

            Args:
                batch (torch.Tensor): Input batch.

            Returns:
                torch.Tensor: Computed feature maps.
            """
            out = self.transpose_conv(batch)
            out = self.batch_norm_layer(out)

            return F.leaky_relu(out)

    def __init__(
        self,
        nn_output_image_res: int,
        latent_dim: int,
        hidden_dims: t.List[int] = [512, 256, 128, 64, 32],
    ) -> None:
        """Constructor.

        Args:
            nn_output_image_res (int): Decoder output image resolution. Assumes square image.
            latent_dim (int): Latend dimension size.
            hidden_dims (t.List[int], optional): Hidden feature map dimensions. Defaults to [512, 256, 128, 64, 32].
        """
        super().__init__()
        self.feature_map_resolution = nn_output_image_res // 2 ** len(hidden_dims)

        self.input_layer = nn.Linear(
            in_features=latent_dim,
            out_features=hidden_dims[0] * self.feature_map_resolution ** 2,
        )

        layers = []
        in_channels = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            transpose_conv_layer = VAEDecoder.TransposeConvolution(
                in_channels=in_channels, out_channels=hidden_dim
            )

            layers.append(transpose_conv_layer)

            in_channels = hidden_dim

        layers.append(
            VAEDecoder.TransposeConvolution(
                in_channels=hidden_dims[-1], out_channels=hidden_dims[-1]
            )
        )
        layers.append(
            nn.Conv2d(
                in_channels=hidden_dims[-1], out_channels=3, kernel_size=3, padding=1
            )
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forwards a batch through the decoder.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Reconstructed images.
        """
        mapped_latten_vector = self.input_layer(batch)

        batch_size = batch.size()[0]
        reshaped_input = mapped_latten_vector.view(
            batch_size, -1, self.feature_map_resolution, self.feature_map_resolution
        )

        output = self.decoder(reshaped_input)
        return torch.sigmoid(output)


class VanillaVAE(nn.Module):
    def __init__(
        self,
        nn_input_image_resolution: int = 512,
        latent_dim: int = 8,
        hidden_dims: t.List[int] = [32, 64, 128, 256, 512],
    ) -> None:
        """Constructor.

        Args:
            nn_input_image_resolution (int): Neural Network input image resolution. Assumes square image. Defaults to 512.
            latent_dim (int): Latend dimension size. Defaults to 8.
            hidden_dims (t.List[int], optional): Feature maps' channels hidden dimensions. Defaults to [32, 64, 128, 256, 512].
        """
        super().__init__()
        self.encoder = VAEEncoder(
            nn_input_image_resolution=nn_input_image_resolution,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        self.decoder = VAEDecoder(
            nn_output_image_res=nn_input_image_resolution,
            latent_dim=latent_dim,
            hidden_dims=list(reversed(hidden_dims)),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forwards batch through VAE.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            torch.Tensor: Reconstructed images.
        """
        encoded_batch = self.encoder(batch)
        batch_reconstructions = self.decoder(encoded_batch)

        return batch_reconstructions
