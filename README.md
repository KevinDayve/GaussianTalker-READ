# GaussianTalker - Dockerisation setup.

This guide documents a robust, Docker-based setup of [GaussianTalker](https://github.com/cvlab-kaist/GaussianTalker) for training and rendering talking-head models using audio. It includes solutions to common pain points such as missing modules, submodules, CUDA issues, and preprocessing failures.

> This guide assumes you're using an Ubuntu EC2 instance with NVIDIA GPU (e.g., G5.X2Large), with [NVIDIA drivers installed](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).

---

## Clone the Repo & Submodules

```bash
cd ~/git
git clone https://github.com/joungbinlee/GaussianTalker.git
cd GaussianTalker
git submodule update --init --recursive
```

# Pull the base CUDA + CuDNN image
docker pull nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Sanity check GPU access
docker run --rm --gpus=all nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 nvidia-smi

# Start a container with GaussianTalker volume-mounted
```bash
docker run -itd \
  --gpus=all \
  --ipc=host \
  --name gaussian_clean_env \
  -v /home/ubuntu/git/GaussianTalker:/root/GaussianTalker \
  nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 \
  /bin/bash
```
# Enter the container
```bash
docker exec -it gaussian_clean_env bash
apt update && apt install -y python3-pip python3-tk ffmpeg libsm6 libxext6 portaudio19-dev
```
# Install Python packages
```
cd /root/GaussianTalker
pip install -r requirements.txt
pip install -e submodules/custom-bg-depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
pip install --upgrade "protobuf<=3.20.1"
```

# Prepare the video from the host EC2 terminal

```bash
mkdir -p ~/git/GaussianTalker/data/obama
cd ~/git/GaussianTalker/data/obama
wget "https://github.com/YudongGuo/AD-NeRF/blob/master/dataset/vids/Obama.mp4?raw=true" -O obama.mp4
```

# Process the video.
```bash
cd /root/GaussianTalker
python3 data_utils/process.py data/obama/obama.mp4
```

### You may also need to run OpenFace Seperately in a dedicated container.
```bash
docker run -it --rm \
  -v /home/ubuntu/git/GaussianTalker:/data \
  --algebr/openface:latest
```
> Inside the container, run:
```bash
cd /data
/home/openface-build/build/bin/FeatureExtraction -f data/obama/obama.mp4
```

# Train the model
```bash
python3 train.py \
  -s data/obama \
  --model_path models/obama \
  --configs arguments/64_dim_1_transformer.py
```
**This will save checkpoints inside models/obama/.**

# Render Output

You can render a preview without training fully by referencing an early checkpoint:
```bash
python3 render.py \
  -s data/obama \
  --model_path models/obama \
  --configs arguments/64_dim_1_transformer.py \
  --iteration 1000 \
  --batch 1 \
  --skip_train --skip_test
```

With a custom audip
```bash
python3 render.py \
  -s data/obama \
  --model_path models/obama \
  --configs arguments/64_dim_1_transformer.py \
  --iteration 10000 \
  --batch 1 \
  --custom_aud aud_novel.npy \
  --custom_wav aud_novel.wav \
  --skip_train --skip_test
```

# Docker Image reference.
```
docker commit gaussian_clean_env gaussian-talker:v2
```
Then run

```bash
docker run -itd \
  --gpus=all \
  --ipc=host \
  --network=host \
  -v /home/ubuntu/git/GaussianTalker:/root/GaussianTalker \
  gaussian-talker:v2 /bin/bash
```


# Final Notes
* Always run training & rendering inside the container
* Always download data on the host, so it's visible to Docker via volume mount
* process.py must be rerun if input is lost or deleted
* Use --break-system-packages if needed for pip in Ubuntu 22.04







