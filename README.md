# NanoSAM

NanoSAM is a [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) model variant that is capable of running in  real-time on NVIDIA Jetson Orin Platforms with NVIDIA TensorRT.  

---
### NOTE

This is the Independent Robotics fork of NanoSAM, the original can be found [here](https://github.com/NVIDIA-AI-IOT/nanosam), and the origninal README can be found in the repo as ORIGINAL_README.md.

The modifications of this repository focus on providing a straightforward interface for use in Independent Robotics's development, with some additional functionality and convenience functions.

---

## Installation
**For all environments, ir_utils must be installed and available.**

### pip install
You can install NanoSAM in most desktop environments, as long as you have a CUDA-capable device set up with [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) and [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).  We will not cover the installation of CUDA or TensorRT, as that's beyond the scope of this repo. However, if you have an environment with CUDA and TensorRT set up, you can operate as follows. 

First, you need to install [Torch2TRT](https://github.com/NVIDIA-AI-IOT/torch2trt?tab=readme-ov-file#setup). Then, you can run the following commands to install NanoSAM.

```bash
# Clone the repository, then change directory to the cloned directory.
pip -r ./requirements.txt
pip install . 
```

### Docker install (Jetson)
Using the Docker install on the Jetson is realtively straightforward. I'm not going to detail the install steps in much detail at this moment because they need to change soon, but I'll briefly explain: You can use the `build_image.bash` script under `docker/jetson/` to build the container, and `run_interactive.bash` to run the Docker container. 

These may not work straightforwardly, because they are currently in a somewhat rough state. The main things that you need to be aware of, for building on the Jetson, are:

1. DOCKER BUILDKIT MUST BE DISABLED. Currently, there is an issue with NVIDIA container toolkit and the modern Docker buildkit, so it needs to be disabled for now.
2. You need a sufficiently large amount of GPU memory for the first time that you run the SAM model, because the system will need to convert an ONXX model to a TensorRT engine, which takes a fair amount of memory. On anything less than a 16GB Nano, mount some swap memory (2-4 GB).


### Docker install (x86_64) 
To use a Docker install on a desktop machine, first you should make sure that you have [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) set up, [Docker](https://docs.docker.com/engine/install/ubuntu/) installed, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. With that done, setting up and using the Docker install is super easy. 

**To Build**: `build_image.bash [local|github]`. You can pass either `local` or `github` as the build target, this will prompt the Dockerfile to either pull Nanosam from your local folder, or from github directly (github credentials aren't needed). **Note:** you do need to update the variables in the script which point to the locations of ir_utils and NanoSAM. 

**To Run (Interactively)**: Simply run `run_interactive.bash`.

**To Run (Example Code)**: Run `run_example.bash [dir_path] [args]`. Following the script, you must pass in a path to a directory for outputs to be written, along with any of the available example flags, which triger various examples to be run. The currently available flags are:

* --points (-p): Point-based example. 
* --bbox (-b): Bounding box-based example. 
* --mask (-m): Mask-based example. 
* --owl (-o): OWL prompt-based example. 
* --sev (-s): Example with several masks (demonstrating drawing functions). 

So, for example, you could run `./run_example.bash ~/docker_out/ -m -b -o`, which should output three images, one for each example, under `~/docker_out/`.


## Programming with NanoSAM (IR interface)
The programming interface provided by this package is localized in the `trt_sam` submodule of `nanosam`. The rest of the repository (forked from NVIDIA, as was the original `trt_sam`), was used to create the NanoSAM models (via knowledge distillation using MobileSAM), that `trt_sam` allows you to run. So, when you are using the SAM interface, you're going to be importing functions or classes from `trt_sam`. 

#### The First Inference
The first time you run NanoSAM (and the first time you use prompt-based prediction), you will need access to the Internet, and it will take a longer time. This is because (due to NVIDIA not making GPU resources available during Docker builds) we wait until the first inference is requested to download and convert critical model files, as the conversion requires GPU access. 

### SAMPredictor
The `SAMPredictor` class provides access to TRT-Engine based SAM outputs. It has 5 functions that you may want to utilize, delinated below:

- **set_image**(img): This function takes an PIL or OpenCV image and sets it to be the current image for prediction.
- **predict_points**(foreground, [background, [iterations]]): This function predicts a mask based on a list of input points that are believed to be in the object(optionally also some background points). The iterations parameter allows multiple prediction steps to refine the mask. 
- **predict_bbox**(bbox, [iterations]): This function predicts a mask based on a bounding box around a known object. The iterations parameter allows multiple prediction steps to refine the mask. 
- **predict_mask**(mask, [iterations]): This function predicts a mask based on an existing mask (possibly created by traditional CV). The iterations parameter allows multiple prediction steps to refine the mask. 
- **predict_prompt**(prompt, [iterations]): This function predicts a mask based on a natural language prompt. The iterations parameter allows multiple prediction steps to refine the mask. 

**NOTE**: All prediction functions take one target, and return a mask, bounding box, and IOU for that one target. Batching targets is not currently supported due to memory constraints. 

### utils
The `utils` submodule provides a number of useful functions, such as functions for calculating bounding boxes and filtering owl detections, but the primary function you're likely to be interested in is:

**markup_image**(image, masks, bboxes, ious, [labels]): This function takes an image, and either a single instance of or a list of masks, bounding boxes and IOUs as returned by SAMPredictor, and optionally labels that match the detections, and returns a nicely marked up image for display.

### Example
The following code demonstrates a minimal example of using a natural-language prompt and mark up the image based on the results.

```python
import numpy as np
import cv2 
from nanosam.trt_sam import SAMPredictor, markup_image

# Instantiate TensorRT predictor
predictor = SAMPredictor()

# Read image and run image encoder
img = cv2.imread(asset_path + "dogs.jpg")
predictor.set_image(img)
prompt = "a dog"

mask, box, iou = predictor.predict_prompt(prompt, iterations=3)

out = markup_image(img, mask, box, iou, prompt)
cv2.imwrite("/home/user/out/sam_prompt.jpg", out)
```