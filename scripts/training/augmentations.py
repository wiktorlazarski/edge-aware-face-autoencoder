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
        "same": A.CropAndPad(percent=0.0, keep_size=False, p=1),
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
        resize_augmentations_keys: t.Optional[t.List[str]] = None,
        augmentations_keys: t.Optional[t.List[str]] = None,
    ) -> None:

        self.resize_augmentations = []
        self.augmentations = []

        if use_all_augmentations or resize_augmentations_keys is None:
            self.resize_augmentations = list(
                AugmentationPipeline.__RESIZE_AUGMENTATIONS.values()
            )
        if use_all_augmentations or augmentations_keys is None:
            self.augmentations = A.Compose(
                list(AugmentationPipeline.__AUGMENTATIONS.values())
            )
        if not use_all_augmentations:
            if resize_augmentations_keys is not None:
                self.resize_augmentations = [
                    AugmentationPipeline.__RESIZE_AUGMENTATIONS[resize_augmentation]
                    for resize_augmentation in resize_augmentations_keys
                ]
            if augmentations_keys is not None:
                self.augmentations = A.Compose(
                    [
                        AugmentationPipeline.__AUGMENTATIONS[augmentation]
                        for augmentation in augmentations_keys
                    ]
                )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if self.resize_augmentations is not None:
            resize_augmentation = random.choice(self.resize_augmentations)
            resized_image = resize_augmentation(image=image)
            augmentation_result = self.augmentations(image=resized_image["image"])
        else:
            augmentation_result = self.augmentations(image=image)
        return augmentation_result["image"]
