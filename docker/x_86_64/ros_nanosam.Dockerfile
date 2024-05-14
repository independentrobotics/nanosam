##############################
# Initial setup
##############################
FROM nvidia/cuda:12.3.2-base-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONWARNINGS="ignore:setup.py install is deprecated::setuptools.command.install"

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt-get install -y curl wget
RUN apt-get install -y build-essential
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio ffmpeg
RUN pip install opencv-python

# This tests to make sure that nvidia-container-toolkit libraries will be available.
# If it fails, make sure that DOCKER_TOOLKIT=0 is set in your environment variables for
# The build, or check the status of the bug that was causing this before
# https://stackoverflow.com/a/75629058
RUN python3 -c "import tensorrt; print(tensorrt.__version__)"

# NanoSAM/NanoOWL setup
WORKDIR /opt

# Install Torch2TRT
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    pip3 install .

# Install Python dependencies for NanoSAM
RUN pip3 install transformers timm matplotlib gdown wget
RUN pip3 install git+https://github.com/openai/CLIP.git

# Install the NanoSAM Python package
RUN git clone https://github.com/independentrobotics/nanosam && \
    cd nanosam && \
    pip3 install .

COPY /ir_utils /root/ir_utils
WORKDIR /root/ir_utils
RUN pip3 install .

WORKDIR /root