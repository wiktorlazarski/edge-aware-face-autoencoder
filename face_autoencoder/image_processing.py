import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class PreprocessingPipeline:
    def __init__(self, nn_image_input_resolution: int) -> None:
        self.nn_image_input_resolution = nn_image_input_resolution
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (nn_image_input_resolution, nn_image_input_resolution)
                ),
            ]
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        return self.preprocess_image(image)

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        return self.image_preprocessing_pipeline(Image.fromarray(image))
