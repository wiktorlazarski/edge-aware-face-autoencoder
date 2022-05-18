import argparse
import logging
import random
import shutil
import typing as t
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset with edges")

    parser.add_argument(
        "--dataset_dir",
        "-d",
        type=Path,
        help="Dataset directory",
    )

    return parser.parse_args()


def create_dataset_with_edges_structure(root_path: Path) -> None:
    logging.info(f"Creating folders structure in {root_path}...")

    train_path = root_path / "train"
    val_path = root_path / "val"
    test_path = root_path / "test"

    for path in [train_path, val_path, test_path]:
        images_path = path / "edges"
        images_path.mkdir(parents=True, exist_ok=False)


def copy_images_with_edges(dset_root: Path) -> None:
    logging.info("Copying images with edges...")
    train = list((dset_root / "train" / "images").glob("*.jpg"))
    val = list((dset_root / "val" / "images").glob("*.jpg"))
    test = list((dset_root / "test" / "images").glob("*.jpg"))
    for name_dataset, dataset in {
        "train": train,
        "val": val,
        "test": test,
    }.items():
        for image in dataset:
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.Canny(img, threshold1=50, threshold2=110)
            cv2.imwrite(
                str(
                    dset_root
                    / name_dataset
                    / "edges"
                    / image.name.replace(".jpg", ".png")
                ),
                img,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info("Building dataset with edges...")
    args = parse_args()

    create_dataset_with_edges_structure(args.dataset_dir)
    copy_images_with_edges(args.dataset_dir)

    logging.info("Finished building dataset with edges.")


if __name__ == "__main__":
    main()
