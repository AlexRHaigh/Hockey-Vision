# Ultralytics' JetPack 6 image (Orin Nano / NX / AGX) has CUDA PyTorch, TensorRT and OpenCV
# built for the Jetson. The repo is mounted at /work at run time, see jetson/README.md.
FROM ultralytics/ultralytics:latest-jetson-jetpack6

RUN pip install --no-cache-dir "lap>=0.5.12"
# timm: only for run_models.py --number-reader temporal (temporal_jersey.py)
RUN pip install --no-cache-dir "timm>=0.9"
# PARSeq model code: only for run_models.py --number-reader parseq (parseq_jersey.py). --no-deps
# skips its training stack (Lightning, Hydra, ...), which inference doesn't need.
RUN pip install --no-cache-dir --no-deps "git+https://github.com/baudm/parseq.git"

WORKDIR /work
