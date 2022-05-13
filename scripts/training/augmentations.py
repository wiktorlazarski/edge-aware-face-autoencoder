import random
import typing as t

import albumentations as A
import numpy as np
import torch


class AugmentationPipeline:
    __RESIZE_AUGMENTATIONS = {
        "crop_0.1": A.CropAndPad(percent=-0.1, keep_size=False, p=1),
        "crop_0.07": A.CropAndPad(percent=-0.07, keep_size=False, p=1),
        "crop_0.04": A.CropAndPad(percent=-0.04, keep_size=False, p=1),
        "crop_0.02": A.CropAndPad(percent=-0.02, keep_size=False, p=1),
        "pad_0.1": A.CropAndPad(percent=0.1, keep_size=False, p=1),
        "pad_0.07": A.CropAndPad(percent=0.07, keep_size=False, p=1),
        "pad_0.04": A.CropAndPad(percent=0.04, keep_size=False, p=1),
        "pad_0.02": A.CropAndPad(percent=0.02, keep_size=False, p=1),
        "identity": A.CropAndPad(percent=0.0, keep_size=False, p=1),
    }

    __AUGMENTATIONS = {
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.3, p=0.5),
        "horizontal_flip": A.HorizontalFlip(p=0.5),
        "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.1),
        "fancyPCA": A.FancyPCA(alpha=0.15, p=0.1),
        "tone_curve": A.RandomToneCurve(scale=0.1, p=0.1),
        "ISO_noise": A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.1),
        "channel_shuffle": A.ChannelShuffle(p=0.05),
        "fog": A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.05
        ),
        "rain": A.RandomRain(
            slant_lower=-8,
            slant_upper=8,
            drop_length=10,
            drop_width=1,
            drop_color=(100, 100, 100),
            blur_value=1,
            brightness_coefficient=0.9,
            rain_type=None,
            p=0.05,
        ),
    }

    def __init__(
        self,
        use_all_augmentations: bool = True,
        resize_augmentation_keys: t.Optional[t.List[str]] = None,
        augmentation_keys: t.Optional[t.List[str]] = None,
    ) -> None:

        self.resize_augmentations = resize_augmentation_keys
        self.augmentations = []

        if use_all_augmentations:
            self.resize_augmentations = list(
                AugmentationPipeline.__RESIZE_AUGMENTATIONS.values()
            )
            self.augmentations = list(AugmentationPipeline.__AUGMENTATIONS.values())
        else:
            if resize_augmentation_keys is not None:
                self.resize_augmentations = [
                    AugmentationPipeline.__RESIZE_AUGMENTATIONS[resize_augmentation]
                    for resize_augmentation in resize_augmentation_keys
                ]
            if augmentation_keys is not None:
                self.augmentations = [
                    AugmentationPipeline.__AUGMENTATIONS[augmentation]
                    for augmentation in augmentation_keys
                ]

        self.augmentations = A.Compose(self.augmentations)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.resize_augmentations is not None:
            resize_augmentation = random.choice(self.resize_augmentations)
            resized_image = resize_augmentation(image=image)
            augmentation_result = self.augmentations(image=resized_image["image"])
        else:
            augmentation_result = self.augmentations(image=image)
        return augmentation_result["image"]


class AugmentationPipelineWithEdges:
    __RESIZE_AUGMENTATIONS = {
        "crop_0.1": A.CropAndPad(percent=-0.1, keep_size=False, p=1),
        "crop_0.07": A.CropAndPad(percent=-0.07, keep_size=False, p=1),
        "crop_0.04": A.CropAndPad(percent=-0.04, keep_size=False, p=1),
        "crop_0.02": A.CropAndPad(percent=-0.02, keep_size=False, p=1),
        "pad_0.1": A.CropAndPad(percent=0.1, keep_size=False, p=1),
        "pad_0.07": A.CropAndPad(percent=0.07, keep_size=False, p=1),
        "pad_0.04": A.CropAndPad(percent=0.04, keep_size=False, p=1),
        "pad_0.02": A.CropAndPad(percent=0.02, keep_size=False, p=1),
        "identity": A.CropAndPad(percent=0.0, keep_size=False, p=1),
    }

    __AUGMENTATIONS = {
        "brightness": A.RandomBrightnessContrast(brightness_limit=0.3, p=0.5),
        "horizontal_flip": A.HorizontalFlip(p=0.5),
        "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.1),
        "fancyPCA": A.FancyPCA(alpha=0.15, p=0.1),
        "tone_curve": A.RandomToneCurve(scale=0.1, p=0.1),
        "ISO_noise": A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.1),
        "channel_shuffle": A.ChannelShuffle(p=0.05),
        "fog": A.RandomFog(
            fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.05
        ),
        "rain": A.RandomRain(
            slant_lower=-8,
            slant_upper=8,
            drop_length=10,
            drop_width=1,
            drop_color=(100, 100, 100),
            blur_value=1,
            brightness_coefficient=0.9,
            rain_type=None,
            p=0.05,
        ),
    }

    def __init__(
        self,
        use_all_augmentations: bool = True,
        resize_augmentation_keys: t.Optional[t.List[str]] = None,
        augmentation_keys: t.Optional[t.List[str]] = None,
    ) -> None:

        self.resize_augmentations = resize_augmentation_keys
        self.augmentations = []

        if use_all_augmentations:
            self.resize_augmentations = list(
                AugmentationPipelineWithEdges.__RESIZE_AUGMENTATIONS.values()
            )
            self.augmentations = list(
                AugmentationPipelineWithEdges.__AUGMENTATIONS.values()
            )
        else:
            if resize_augmentation_keys is not None:
                self.resize_augmentations = [
                    AugmentationPipelineWithEdges.__RESIZE_AUGMENTATIONS[
                        resize_augmentation
                    ]
                    for resize_augmentation in resize_augmentation_keys
                ]
            if augmentation_keys is not None:
                self.augmentations = [
                    AugmentationPipelineWithEdges.__AUGMENTATIONS[augmentation]
                    for augmentation in augmentation_keys
                ]

        self.augmentations = A.Compose(self.augmentations)

    def __call__(
        self, image: np.ndarray, edge: np.ndarray
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if self.resize_augmentations is not None:
            resize_augmentation = random.choice(self.resize_augmentations)
            resized_image = resize_augmentation(image=image, mask=edge)
            augmentation_result = self.augmentations(
                image=resized_image["image"], mask=resized_image["mask"]
            )
        else:
            augmentation_result = self.augmentations(image=image, mask=edge)
        return augmentation_result["image"], augmentation_result["mask"]
