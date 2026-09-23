import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--half', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.weights)
    model.export(
        format='engine',
        imgsz=args.imgsz,
        half=args.half,
        device=0,
    )
