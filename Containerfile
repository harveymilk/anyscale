# Containerfile -- build with "anyscale image build -f Containerfile -n video-clip-summarizer-env --ray-version 2.33.0"
FROM anyscale/ray:2.33.0-py310-cu123

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1


RUN python -m pip install -U pip setuptools wheel

# 2) Install CUDA wheels for PyTorch first (cu121 wheels run fine on CUDA 12.x)
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.2 torchvision==0.17.2


RUN python -m pip install \
    "ray[data]==2.33.0" \
    opencv-python-headless==4.9.0.80 \
    imageio==2.34.0 \
    imageio-ffmpeg==0.4.9 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    fsspec==2024.6.0 \
    s3fs==2024.6.0 \
    dotenv==1.1.0

WORKDIR /workspace
