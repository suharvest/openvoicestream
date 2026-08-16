# GPU wheel drop-in (Jetson)

Place the Jetson aarch64 `onnxruntime-gpu` wheel here before building to get
a GPU image (TensorRT + CUDA EPs); leave empty for the CPU image (x86/CI).

Proven combo (JetPack 6, cp310): `onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl`
(kept on the Jetson at /home/harvest/wheels/ — ~250MB, not committed to git).

Build GPU (fails loudly if the wheel is missing):

    docker build -f agent/Dockerfile.rebot-arm --build-arg REQUIRE_GPU=1 -t voice-rebot-arm:oneclick-gpu-YYYYMMDD .

Runtime still needs the host JetPack CUDA/TensorRT/cuDNN libs mounted by the
gpu-libs-init service in the compose stack (the wheel links against them).
