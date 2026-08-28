"""Numerically diff every whisper-base encoder plan against onnxruntime.
A TRT engine that builds and runs is not necessarily a correct engine — the
fp16 build of this graph scores cosine 0.81 while looking entirely healthy."""
import sys, glob, wave
import numpy as np, tensorrt as trt, onnxruntime as ort
from cuda import cudart
sys.path.insert(0, "/home/harvest/whisper-bench")
from trt_whisper_run import log_mel

L = trt.Logger(trt.Logger.ERROR)
filters = np.loadtxt("/home/harvest/whisper-bench/model/mel_80_filters.txt",
                     dtype=np.float32).reshape((80, 201))
with wave.open("/home/harvest/whisper-bench/corpus/short/en_short_01.wav", "rb") as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
mel = np.ascontiguousarray(log_mel(pcm.astype(np.float32) / 32768.0, filters, 3000)[None, :, :])

s = ort.InferenceSession("enc_base_30s.onnx", providers=["CPUExecutionProvider"])
ref = s.run(None, {s.get_inputs()[0].name: mel})[0]
b = ref.ravel().astype(np.float64)

print(f"{'plan':<34}{'cosine':>11}{'maxabsdiff':>12}{'std':>9}  (ref std %.4f)" % ref.std())
for plan in sorted(glob.glob("enc_base_30s*.plan")):
    e = trt.Runtime(L).deserialize_cuda_engine(open(plan, "rb").read())
    c = e.create_execution_context()
    iname, oname = e.get_tensor_name(0), e.get_tensor_name(1)
    c.set_input_shape(iname, mel.shape)
    out = np.empty(tuple(c.get_tensor_shape(oname)), dtype=np.float32)
    di = cudart.cudaMalloc(mel.nbytes)[1]; do = cudart.cudaMalloc(out.nbytes)[1]
    cudart.cudaMemcpy(di, mel.ctypes.data, mel.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    c.set_tensor_address(iname, int(di)); c.set_tensor_address(oname, int(do))
    st = cudart.cudaStreamCreate()[1]
    c.execute_async_v3(stream_handle=st); cudart.cudaStreamSynchronize(st)
    cudart.cudaMemcpy(out.ctypes.data, do, out.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    a = out.ravel().astype(np.float64)
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    flag = "OK" if cos >= 0.999 else "BROKEN"
    print(f"{plan:<34}{cos:>11.6f}{np.abs(a-b).max():>12.3f}{out.std():>9.4f}  {flag}")
    cudart.cudaFree(di); cudart.cudaFree(do)
