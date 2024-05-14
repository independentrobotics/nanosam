# This file provides utility functions for the creation of trt egnines, 
# so that it can be easily integrated into the setup.py install process.
from ir_utils.filesystem_tools import get_dl_model_directory, create_dl_model_directory
import os
import gdown
import wget
import subprocess

def setup():
    try:
        model_path = get_dl_model_directory("nanosam")
    except ValueError:
        create_dl_model_directory("nanosam")

    # Set up tmp directory (technically we should check for engine existance before this but eh.)
    prev_wd = os.getcwd()
    os.chdir(model_path)
    os.mkdir("./tmp")

    create_image_encoder_engine(model_path)
    create_mask_decoder_engine(model_path)

    # Cleanup process
    os.rmdir('./tmp')
    os.chdir(prev_wd)

def create_image_encoder_engine(model_path):

    # First, check engine existance.
    engine_name = "resnet18_image_encoder.engine"
    if os.path.exists(model_path + engine_name):
        print(f"Image encoder engine exists under model config directory {model_path}, skipping")
        return False

    # Download ONNX model
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR"
    onnx_path = "./tmp/resnet18_image_encoder.onnx"
    gdown.download(url, onnx_path)

    # Convert ONNX to TRT Engine.
    print("Converting ONNX to TRTEngine...")
    base_command = "/usr/src/tensorrt/bin/trtexec"
    onnx = f"--onnx={onnx_path}"
    engine = f"--saveEngine=./{engine_name}"
    precision = "--fp16"

    command = ' '.join(base_command, onnx, engine, precision)
    subprocess.run(command,shell=True)
    
def create_mask_decoder_engine(model_path):
    # First, check engine existance.
    engine_name = "mobile_sam_mask_decoder.engine"
    if os.path.exists(model_path + engine_name):
        print(f"Mask decoder engine exists under model config directory {model_path}, skipping")
        return False

    print("Downloading model...")
    url = "https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx"
    onxx_path = "./tmp/mobile_sam_mask_decoder.onnx"
    wget.download(url, out=onxx_path)

    # Convert ONNX to TRT Engine.
    print("Converting ONNX to TRTEngine...")
    base_command = "/usr/src/tensorrt/bin/trtexec"
    onnx = f"--onnx={onxx_path}"
    engine = f"--saveEngine=./{engine_name}"
    shapes = "--minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10"

    command = ' '.join(base_command, onnx, engine, shapes)
    subprocess.run(command,shell=True)