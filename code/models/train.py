import argparse
from ultralytics import YOLO

# Register TBAM so that the YAML parser can find it.
import ultralytics.nn.tasks as tasks
from models.tbam import TBAM

tasks.TBAM = TBAM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/data.yaml')
    parser.add_argument('--cfg', type=str, default='models/yolov8_tbam_emsffn.yaml')
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr0', type=float, default=0.05)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--device', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.cfg)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        pretrained=args.weights,
    )
