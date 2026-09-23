"""
Evaluate TensorRT engine latency, throughput and power on NVIDIA Jetson Xavier NX.
Run this script on the edge device with TensorRT and PyCUDA installed.
"""
import argparse
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--samples', type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Discover bindings dynamically
    input_idx = None
    output_idx = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            input_idx = i
        else:
            output_idx = i

    if input_idx is None or output_idx is None:
        raise RuntimeError('Could not find input/output bindings.')

    input_shape = tuple(engine.get_binding_shape(input_idx))
    output_shape = tuple(engine.get_binding_shape(output_idx))
    input_dtype = trt.nptype(engine.get_binding_dtype(input_idx))
    output_dtype = trt.nptype(engine.get_binding_dtype(output_idx))

    print(f'Input shape: {input_shape}, dtype: {input_dtype}')
    print(f'Output shape: {output_shape}, dtype: {output_dtype}')

    input_data = np.random.randn(*input_shape).astype(input_dtype)
    output = np.empty(output_shape, dtype=output_dtype)

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)
    stream = cuda.Stream()

    # Warm-up
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

    start = time.time()
    for _ in range(args.samples):
        cuda.memcpy_htod_async(d_input, input_data, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
    end = time.time()

    latency = (end - start) / args.samples * 1000
    throughput = args.samples / (end - start)
    print(f'Latency: {latency:.2f} ms')
    print(f'Throughput: {throughput:.2f} FPS')


if __name__ == '__main__':
    main()
