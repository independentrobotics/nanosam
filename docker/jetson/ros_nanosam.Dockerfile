##############################
# Initial setup
##############################
FROM dustynv/l4t-ml:r36.2.0
SHELL ["/bin/bash", "-c"]
ENV PYTHONWARNINGS="ignore:setup.py install is deprecated::setuptools.command.install"

# This tests to make sure that nvidia-container-toolkit libraries will be available.
# If it fails, make sure that DOCKER_TOOLKIT=0 is set in your environment variables for
# The build, or check the status of the bug that was causing this before
# https://stackoverflow.com/a/75629058
RUN python3 -c "import tensorrt; print(tensorrt.__version__)"

##############################
# ROS Humble Install
##############################
RUN apt update
RUN apt install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

RUN apt install -y software-properties-common
RUN add-apt-repository universe
RUN apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get purge -y '*opencv*'
RUN apt update
RUN apt upgrade -y
RUN apt install ros-humble-ros-base -y
RUN rm -r /usr/local/lib/python3.10/dist-packages/cv2

RUN python3 -m pip install opencv-python
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y build-essential
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libgeographic-dev
RUN apt-get install -y libopencv-dev
RUN apt-get install -y ros-humble-geographic-msgs
RUN apt-get install -y ros-humble-angles
RUN apt-get install -y ros-humble-diagnostic-updater
RUN apt-get install -y ros-humble-geodesy
RUN apt-get install -y ros-humble-camera-info-manager
RUN apt-get install -y ros-humble-image-transport
RUN apt-get install -y ros-humble-image-transport-plugins
RUN apt-get install -y ros-humble-cv-bridge
RUN apt-get install -y ros-humble-rosbag2-storage-mcap
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio ffmpeg
RUN python3 -m pip install pyserial transforms3d pyubx2 colcon-common-extensions albumentations

##############################
# NanoSAM/NanoOWL setup
#
# Install instructions for these largely come from github.com/dusty-nv/jetson-containers
##############################
WORKDIR /opt

# Install Python dependencies for NanoSAM
run pip3 install transformers timm matplotlib gdown
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN python3 -c "from transformers import (OwlViTProcessor, OwlViTForObjectDetection); \
                OwlViTProcessor.from_pretrained(\"google/owlvit-base-patch32\");  \
                OwlViTForObjectDetection.from_pretrained(\"google/owlvit-base-patch32\");"

# Install Torch2TRT
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install

ARG CHEAT=unknown
# Install the NanoSAM Python package
RUN git clone https://github.com/independentrobotics/nanosam && \
    cd nanosam && \
    python3 setup.py develop --user

# Build the TensorRT engine for the mask decoder
RUN mkdir /opt/nanosam/data && \
    wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate \
	 https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx \
	 -O /opt/nanosam/data/mobile_sam_mask_decoder.onnx

RUN cd /opt/nanosam && \
    /usr/src/tensorrt/bin/trtexec \
        --onnx=data/mobile_sam_mask_decoder.onnx \
        --saveEngine=data/mobile_sam_mask_decoder.engine \
        --minShapes=point_coords:1x1x2,point_labels:1x1 \
        --optShapes=point_coords:1x1x2,point_labels:1x1 \
        --maxShapes=point_coords:1x10x2,point_labels:1x10

# Build the TensorRT engine for the NanoSAM image encoder
RUN cd /opt/nanosam/data/ && \
    gdown https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR && \
    ls -lh && \
    cd /opt/nanosam/ && \
    /usr/src/tensorrt/bin/trtexec \
        --onnx=data/resnet18_image_encoder.onnx \
        --saveEngine=data/resnet18_image_encoder.engine \
        --fp16

##############################
# Final setup for IR Docker systems (uncomment if you're using this container as part of a IR Docker setup. )
##############################
# COPY /ir_utils /root/ir_utils
# WORKDIR /root/ir_utils
# RUN pip3 install .

# COPY /core_ws /root/core_ws
# WORKDIR /root/core_ws
# RUN . /opt/ros/humble/setup.sh && colcon build
# RUN . install/setup.bash

WORKDIR /root