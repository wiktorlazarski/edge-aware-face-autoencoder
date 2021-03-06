import pathlib as p
import typing as t

import cv2
import torch
import torchvision

import face_autoencoder.image_processing as ip
import scripts.training.augmentations as aug


class CelebAFaceAutoencoderDataset(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset_root: str,
        preprocess_pipeline: t.Optional[ip.PreprocessingPipeline] = None,
        augmentation_pipeline: t.Optional[aug.AugmentationPipeline] = None,
    ) -> None:
        super().__init__(root=dataset_root)

        self.images = self._load_images()
        self.preprocess_pipeline = preprocess_pipeline
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.images[index]
        image = self._load_sample(image)

        if self.augmentation_pipeline is not None:
            image = self.augmentation_pipeline(image=image)

        if self.preprocess_pipeline is not None:
            image = self.preprocess_pipeline.preprocess_image(image=image)

        return image

    def _load_images(self) -> t.List[str]:
        images_path = p.Path(self.root) / "images"

        images = sorted(list(images_path.glob("*.jpg")))

        return images

    def _load_sample(self, image_name: p.Path) -> torch.Tensor:
        image = cv2.imread(str(image_name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image


class CelebAFaceAutoencoderDatasetWithEdges(torchvision.datasets.VisionDataset):
    def __init__(
        self,
        dataset_root: str,
        preprocess_pipeline: t.Optional[ip.PreprocessingPipeline] = None,
        augmentation_pipeline: t.Optional[aug.AugmentationPipeline] = None,
    ) -> None:
        super().__init__(root=dataset_root)

        self.images, self.edges = self._load_samples()
        self.preprocess_pipeline = preprocess_pipeline
        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        im_edges = self.edges[index]
        image, edges = self._load_sample(image, im_edges)

        if self.augmentation_pipeline is not None:
            image, edges = self.augmentation_pipeline(image=image, edges=edges)

        if self.preprocess_pipeline is not None:
            image = self.preprocess_pipeline.preprocess_image(image=image)
            edges = self.preprocess_pipeline.preprocess_edge(edges=edges)

        return image, edges

    def _load_samples(self) -> t.List[str]:
        images_path = p.Path(self.root) / "images"
        edges_path = p.Path(self.root) / "edges"

        images = sorted(list(images_path.glob("*.jpg")))
        edges = sorted(list(edges_path.glob("*.png")))

        return images, edges

    def _load_sample(self, image_name: p.Path, edge_name: p.Path) -> torch.Tensor:
        image = cv2.imread(str(image_name), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        edge = cv2.imread(str(edge_name), cv2.IMREAD_GRAYSCALE)

        return image, edge
