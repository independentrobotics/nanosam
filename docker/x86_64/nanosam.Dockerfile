##############################
# Initial setup
##############################
# If you run into issues with CUDA version, you should switch to a different CUDA image.
FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu22.04 as base
SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update
RUN apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt-get install -y curl wget git build-essential
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio ffmpeg
RUN apt-get install -y libnvinfer-bin
RUN pip install opencv-python

# NanoSAM setup
WORKDIR /opt/

# Install Torch2TRT
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 -m pip install -r requirements/requirements_10.txt && \
    python3 -m pip install .

# Install Python dependencies for NanoSAM
RUN python3 -m pip install transformers timm matplotlib gdown wget scipy
RUN python3 -m pip install git+https://github.com/openai/CLIP.git

# Install the NanoSAM Python package

COPY --from=ir_utils . /opt/ir_utils
RUN cd ir_utils && \
    python3 -m pip install .


#################################
# Local version: We copy from local directories.
#################################
FROM base as local
# # Grab most recent from build directory.
COPY --from=nanosam . /opt/nanosam
RUN cd nanosam && \ 
    python3 -m pip install .  

RUN cp /opt/nanosam/examples/basic_usage.py /root/example.py
WORKDIR /root
#################################
# Github version: We clone from Github
#################################
FROM base as github

# Pull most recent from Github
RUN git clone https://github.com/independentrobotics/nanosam.git && \
    cd nanosam && \
    python3 setup.py install

RUN cp /opt/nanosam/examples/basic_usage.py /root/example.py
WORKDIR /root