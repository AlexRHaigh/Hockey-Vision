# Ultralytics' JetPack 6 image (Orin Nano / NX / AGX) has CUDA PyTorch, TensorRT and OpenCV
# built for the Jetson. The repo is mounted at /work at run time, see jetson/README.md.
FROM ultralytics/ultralytics:latest-jetson-jetpack6

RUN pip install --no-cache-dir "lap>=0.5.12"

WORKDIR /work
