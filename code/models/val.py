import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, default='../data/data.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(metrics)
