# Edge Deployment on NVIDIA Jetson Xavier NX

## Hardware

- NVIDIA Jetson Xavier NX
- TensorRT FP16
- Input resolution: 640 × 640
- Number of samples: 500
- Average of multiple inferences

## Steps

1. Install JetPack with TensorRT and PyCUDA.
2. Copy the trained `best.pt` to the Jetson.
3. Export TensorRT engine on the Jetson:

```bash
python export_edge.py --weights ../weights/best.pt --imgsz 640 --half
