import typing as t

import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from torchvision import transforms


class PreprocessingPipeline:
    def __init__(
        self, nn_input_image_resolution: int, edge_weight: t.Optional[int] = None
    ) -> None:
        self.nn_input_image_resolution = nn_input_image_resolution
        self.edge_weight = edge_weight
        self.image_preprocessing_pipeline = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (nn_input_image_resolution, nn_input_image_resolution)
                ),
            ]
        )

    def __call__(
        self, image: np.ndarray, edges: t.Optional[np.ndarray] = None
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        preprocessed_img = self.preprocess_image(image)

        if edges is not None:
            preprocessed_edges = self.preprocess_edge(edges)

        return (
            preprocessed_img,
            preprocessed_edges if edges is not None else preprocessed_img,
        )

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        return self.image_preprocessing_pipeline(Image.fromarray(image))

    def preprocess_edge(self, edges: np.ndarray) -> torch.Tensor:
        edges = resize(
            edges,
            (self.nn_input_image_resolution, self.nn_input_image_resolution),
            mode="edge",
            anti_aliasing=False,
            anti_aliasing_sigma=None,
            preserve_range=True,
            order=0,
        )
        edges = torch.Tensor(edges)
        edges[edges == 255] = self.edge_weight
        edges[edges == 0] = 1

        return edges
