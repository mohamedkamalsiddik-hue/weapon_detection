import argparse
import logging
import random
import shutil
from pathlib import Path

import cv2
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a YOLOv8 dataset directory and optionally convert images to grayscale."
    )
    parser.add_argument("--source", required=True, type=Path, help="Path to the source dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset root path.")
    parser.add_argument("--split", nargs=3, type=float, default=[0.7, 0.15, 0.15], help="Train/val/test split ratios.")
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale-compatible BGR output.")
    parser.add_argument("--classes", type=Path, default=None, help="Optional class names file with one name per line.")
    parser.add_argument("--yaml", type=Path, default=None, help="Optional output path for the generated YOLO data YAML file.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def find_dirs(root: Path):
    images_dir = None
    labels_dir = None
    for sub in [root, root / "images", root / "imgs"]:
        if sub.is_dir():
            for child in sub.iterdir():
                if child.is_dir() and child.name.lower() in {"images", "imgs", "images/train", "images/val", "images/test"}:
                    images_dir = sub
                    break
            if images_dir:
                break
    if (root / "images").is_dir() and (root / "labels").is_dir():
        images_dir = root / "images"
        labels_dir = root / "labels"
    elif (root / "images").is_dir() and (root / "labels").exists():
        labels_dir = root / "labels"
    elif (root / "labels").is_dir() and (root / "images").exists():
        images_dir = root / "images"
        labels_dir = root / "labels"
    elif (root / "images").is_dir():
        images_dir = root / "images"
        labels_dir = root / "labels"
    if labels_dir is None:
        labels_dir = root / "labels"
    if images_dir is None:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() in {"images", "imgs"}:
                images_dir = child
    if labels_dir is None:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == "labels":
                labels_dir = child
    if images_dir is None or labels_dir is None:
        raise FileNotFoundError("Could not locate images and labels directories in source path.")
    return images_dir, labels_dir


def collect_image_files(images_dir: Path):
    files = [p for p in sorted(images_dir.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    return files


def read_classes(classes_path: Path):
    if classes_path and classes_path.exists():
        return [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
    return []


def extract_label_classes(label_paths):
    class_ids = set()
    for label_path in label_paths:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(parts[0]))
                    except ValueError:
                        continue
    return sorted(class_ids)


def prepare_split(image_files, split_ratios):
    random.shuffle(image_files)
    train_ratio, val_ratio, test_ratio = split_ratios
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train : n_train + n_val]
    test_files = image_files[n_train + n_val :]
    return train_files, val_files, test_files


def convert_image(source_path: Path, target_path: Path, grayscale: bool):
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {source_path}")
    if grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_path), image)


def copy_label(source_path: Path, target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(target_path))


def build_dataset_split(split_name, files, images_dir, labels_dir, output_root, grayscale):
    images_out = output_root / "images" / split_name
    labels_out = output_root / "labels" / split_name
    written_labels = []
    for src_image in files:
        relative = src_image.relative_to(images_dir)
        dst_image = images_out / relative
        label_file = labels_dir / relative.with_suffix(".txt").name
        if not label_file.exists():
            logging.warning("Skipping image without label: %s", src_image)
            continue
        dst_label = labels_out / label_file.name
        convert_image(src_image, dst_image, grayscale)
        copy_label(label_file, dst_label)
        written_labels.append(dst_label)
    return written_labels


def generate_data_yaml(output_root: Path, classes, yaml_path: Path):
    if not classes:
        raise ValueError("Class names are required to write YOLO data YAML.")
    data = {
        "train": str((output_root / "images" / "train").resolve()),
        "val": str((output_root / "images" / "val").resolve()),
        "test": str((output_root / "images" / "test").resolve()),
        "nc": len(classes),
        "names": classes,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    logging.info("Wrote data YAML to %s", yaml_path)


def main():
    args = parse_args()
    setup_logging()
    random.seed(42)

    images_dir, labels_dir = find_dirs(args.source)
    image_files = collect_image_files(images_dir)
    train_files, val_files, test_files = prepare_split(image_files, args.split)

    logging.info("Source images: %s", images_dir)
    logging.info("Source labels: %s", labels_dir)
    logging.info("Output root: %s", args.output)
    logging.info("Split sizes: train=%d val=%d test=%d", len(train_files), len(val_files), len(test_files))

    written_labels = []
    written_labels += build_dataset_split("train", train_files, images_dir, labels_dir, args.output, args.grayscale)
    written_labels += build_dataset_split("val", val_files, images_dir, labels_dir, args.output, args.grayscale)
    written_labels += build_dataset_split("test", test_files, images_dir, labels_dir, args.output, args.grayscale)

    classes = read_classes(args.classes)
    if not classes:
        ids = extract_label_classes(written_labels)
        classes = [f"class_{i}" for i in ids]
        logging.info("Generated default class names from labels: %s", classes)

    yaml_path = args.yaml or args.output / "data.yaml"
    generate_data_yaml(args.output, classes, yaml_path)

    logging.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
