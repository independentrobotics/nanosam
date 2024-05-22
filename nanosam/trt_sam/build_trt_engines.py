# This file provides utility functions for the creation of trt egnines, 
# so that it can be easily integrated into the setup.py install process.
from ir_utils.filesystem_tools import get_dl_model_directory, create_dl_model_directory
import os, shutil
import gdown
import wget
import subprocess

def build_engines():
    try:
        model_path = get_dl_model_directory("nanosam")
    except ValueError:
        create_dl_model_directory("nanosam")
        model_path = get_dl_model_directory("nanosam")

    prev_wd = os.getcwd()
    os.chdir(model_path)

    create_image_encoder_engine(model_path)
    create_mask_decoder_engine(model_path)

    # Cleanup process
    os.chdir(prev_wd)

def create_image_encoder_engine(model_path):

    # First, check engine existance.
    engine_name = "resnet18_image_encoder.engine"
    if os.path.exists(model_path + engine_name):
        print(f"Image encoder engine exists under model config directory {model_path}, skipping")
        return False

    # Download ONNX model
    onnx_path = "./resnet18_image_encoder.onnx"
    if not os.path.exists(onnx_path):
        print("Downloading model...")
        url = "https://drive.google.com/uc?id=14-SsvoaTl-esC3JOzomHDnI9OGgdO2OR"
        gdown.download(url, onnx_path)

    # Convert ONNX to TRT Engine.
    print("Converting ONNX to TRTEngine...")
    base_command = "/usr/src/tensorrt/bin/trtexec"
    onnx = f"--onnx={onnx_path}"
    engine = f"--saveEngine=./{engine_name}"
    precision = "--fp16"

    command = ' '.join([base_command, onnx, engine, precision])
    subprocess.run(command,shell=True)
    
def create_mask_decoder_engine(model_path):
    # First, check engine existance.
    engine_name = "mobile_sam_mask_decoder.engine"
    if os.path.exists(model_path + engine_name):
        print(f"Mask decoder engine exists under model config directory {model_path}, skipping")
        return False

    onnx_path = "./mobile_sam_mask_decoder.onnx"
    if not os.path.exists(onnx_path):
        print("Downloading model...")
        url = "https://nvidia.box.com/shared/static/ho09o7ohgp7lsqe0tcxqu5gs2ddojbis.onnx"
        wget.download(url, out=onnx_path)

    # Convert ONNX to TRT Engine.
    print("Converting ONNX to TRTEngine...")
    base_command = "/usr/src/tensorrt/bin/trtexec"
    onnx = f"--onnx={onnx_path}"
    engine = f"--saveEngine=./{engine_name}"
    shapes = "--minShapes=point_coords:1x1x2,point_labels:1x1 --optShapes=point_coords:1x1x2,point_labels:1x1 --maxShapes=point_coords:1x10x2,point_labels:1x10"

    command = ' '.join([base_command, onnx, engine, shapes])
    subprocess.run(command,shell=True)

if __name__ == "__main__":
    build_engines()