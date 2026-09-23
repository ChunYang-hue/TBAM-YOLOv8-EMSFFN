# Model Weights

Place the trained model weights here:

- `best.pt`: best validation checkpoint.
- `last.pt`: last epoch checkpoint.
- `best.engine`: TensorRT FP16 engine exported on NVIDIA Jetson Xavier NX.

> Before uploading, confirm that the institution permits public release of model weights.  
> TensorRT engine is device-specific and may be omitted; users can export it themselves following `code/configs/edge_jetson_xavier_nx.md`.
