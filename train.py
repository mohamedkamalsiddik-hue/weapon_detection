import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for weapon detection.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the YOLO data YAML file.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 pretrained model or custom weights file.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Project directory for training results.")
    parser.add_argument("--name", type=str, default="weapon_detection", help="Subfolder name for this training run.")
    parser.add_argument("--device", type=str, default="0", help="Compute device, e.g. 0 for GPU or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        project=str(args.project),
        name=args.name,
        device=args.device,
        save=True,
    )

    print(f"Training complete. Results saved to {args.project / args.name}")


if __name__ == "__main__":
    main()
