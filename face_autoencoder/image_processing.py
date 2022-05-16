import typing as t

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PreprocessingPipeline:
    def __init__(self, nn_input_image_resolution: int) -> None:
        self.nn_input_image_resolution = nn_input_image_resolution
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (nn_input_image_resolution, nn_input_image_resolution)
                ),
            ]
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.preprocess_image(image)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        return self.image_preprocessing_pipeline(Image.fromarray(image))


class PreprocessingPipelineWithEdges:
    def __init__(self, nn_input_image_resolution: int, weights: int) -> None:
        self.nn_input_image_resolution = nn_input_image_resolution
        self.weights = weights
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (nn_input_image_resolution, nn_input_image_resolution)
                ),
            ]
        )

    def __call__(
        self, image: np.ndarray, edge: np.ndarray
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess_image(image), self.preprocess_edge(edge)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        return self.image_preprocessing_pipeline(Image.fromarray(image))

    def preprocess_edge(self, edge: np.ndarray) -> torch.Tensor:
        edge = self.image_preprocessing_pipeline(Image.fromarray(edge))
        edge[edge != 0] = self.weights
        edge[edge == 0] = 1

        return edge
