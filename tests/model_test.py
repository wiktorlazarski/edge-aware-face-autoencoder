import pytest
import torch

import face_autoencoder.model as mdl


def test_encoder_init_raise_error() -> None:
    # given
    nn_input_image_resolution = 4
    hidden_dims = [32, 64, 128, 256, 512]

    # when / then
    with pytest.raises(mdl.VAEEncoderError):
        _ = mdl.VAEEncoder(
            latent_dim=8,
            nn_input_image_resolution=nn_input_image_resolution,
            hidden_dims=hidden_dims,
        )


def test_encoder_output_size() -> None:
    # given
    latent_dim = 8
    nn_input_image_resolution = 256
    hidden_dims = [32, 64, 128, 256, 512]
    encoder = mdl.VAEEncoder(
        latent_dim=latent_dim,
        nn_input_image_resolution=nn_input_image_resolution,
        hidden_dims=hidden_dims,
    )

    batch_size = 10
    mock_batch = torch.ones(
        (batch_size, 3, nn_input_image_resolution, nn_input_image_resolution)
    )

    expected_feature_maps_size = torch.Size((batch_size, latent_dim))
    expected_mu_size = torch.Size((batch_size, latent_dim))
    expected_log_var_size = torch.Size((batch_size, latent_dim))

    # when
    with torch.no_grad():
        feature_maps, mu, log_var = encoder.forward(mock_batch)

    # then
    assert feature_maps.size() == expected_feature_maps_size
    assert mu.size() == expected_mu_size
    assert log_var.size() == expected_log_var_size


def test_decoder_output_size() -> None:
    # given
    nn_output_image_res = 256
    latent_dim = 8
    hidden_dims = [512, 256, 128, 64, 32]

    decoder = mdl.VAEDecoder(
        nn_output_image_res=nn_output_image_res,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    batch_size = 4
    mock_batch = torch.ones((batch_size, latent_dim))

    expected_result = torch.Size(
        (batch_size, 3, nn_output_image_res, nn_output_image_res)
    )

    # when
    with torch.no_grad():
        result = decoder.forward(mock_batch)

    # then
    assert result.size() == expected_result


def test_variational_autoencoder_output_size() -> None:
    # given
    nn_input_image_resolution = 256
    latent_dim = 8
    hidden_dims = [32, 64, 128, 256, 512]

    vae = mdl.VanillaVAE(
        nn_input_image_resolution=nn_input_image_resolution,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )

    batch_size = 4
    mock_batch = torch.ones(
        (batch_size, 3, nn_input_image_resolution, nn_input_image_resolution)
    )

    expected_feature_maps_size = mock_batch.size()
    expected_mu_size = torch.Size((batch_size, latent_dim))
    expected_log_var_size = torch.Size((batch_size, latent_dim))

    # when
    with torch.no_grad():
        feature_maps, mu, log_var = vae.forward(mock_batch)

    # then
    assert feature_maps.size() == expected_feature_maps_size
    assert mu.size() == expected_mu_size
    assert log_var.size() == expected_log_var_size
