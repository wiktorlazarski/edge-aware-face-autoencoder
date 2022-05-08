import os
import typing as t

import gdown
import numpy as np
import torch
from loguru import logger

import face_autoencoder.constants as C
import face_autoencoder.image_processing as ip
import face_autoencoder.model as mdl


class VAEReconstructionPipeline:
    def __init__(
        self,
        model_path: str = C.FACE_VAE_MODEL_PATH,
        model_url: str = C.FACE_VAE_MODEL_DOWNLOAD_URL,
    ):
        if not os.path.exists(model_path):
            logger.warning(
                f"Model {model_path} doesn't exist. Downloading and caching the model to {model_url}."
            )

            model_path = C.FACE_VAE_MODEL_PATH
            gdown.download(model_url, model_path, quiet=False)

        ckpt = torch.load(model_path, map_location=torch.device("cpu"))
        hparams = ckpt["hyper_parameters"]

        self._preprocessing_pipeline = ip.PreprocessingPipeline(
            nn_input_image_resolution=hparams["nn_input_image_resolution"]
        )
        self.nn_model = mdl.VanillaVAE.load_from_checkpoint(ckpt_path=model_path)
        self.nn_model.eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.reconstruct(image)

    def reconstruct(self, image: np.ndarray) -> np.ndarray:
        preprocessed_in = self._preprocess(image)

        model_reconstrution, _, _ = self.nn_model(preprocessed_in)

        return self._postprocess(model_reconstrution)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        preprocessed_image = self._preprocessing_pipeline.preprocess_image(image)
        preprocessed_image = preprocessed_image.unsqueeze(0)

        return preprocessed_image

    def _postprocess(self, model_out: t.Any) -> t.Any:
        out = model_out.squeeze()
        out = out.permute(1, 2, 0)
        out = out.detach().numpy() * 255

        return out.astype(np.uint8)
