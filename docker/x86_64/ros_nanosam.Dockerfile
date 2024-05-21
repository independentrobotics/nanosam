##############################
# Initial setup
##############################
FROM nvidia/cuda:12.3.2-base-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# ENV PYTHONWARNINGS="ignore:setup.py install is deprecated:p:setuptools.command.install"

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt-get install -y curl wget git build-essential
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio ffmpeg
RUN pip install opencv-python

# NanoSAM setup
WORKDIR /root

# Install PyTorch
RUN python3 -m pip install torch torchvision

# Install Tensorrt
RUN python3 -m pip install --pre --upgrade tensorrt
RUN apt-get install -y libnvinfer-bin

# Install Torch2TRT
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 -m pip install -r requirements/requirements_10.txt && \
    python3 -m pip install .

# Install Python dependencies for NanoSAM
RUN python3 -m pip install transformers timm matplotlib gdown wget
RUN python3 -m pip install git+https://github.com/openai/CLIP.git

# Install the NanoSAM Python package
RUN git clone https://github.com/independentrobotics/nanosam.git && \
    cd nanosam && \
    python3 setup.py install

COPY ir_utils /root/ir_utils
RUN cd ir_utils && \
    pip3 install .
