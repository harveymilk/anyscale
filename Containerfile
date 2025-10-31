# Containerfile
FROM anyscale/ray:2.33.0-py310-cu123

# (optional) video utils — remove if you don’t need ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt

# Install “regular” deps first
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install CUDA wheels for PyTorch 2.2.2 (cu121 wheels work with CUDA 12.3 drivers)
RUN pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

# Copy your repo last (better layer caching)
COPY . /workspace
