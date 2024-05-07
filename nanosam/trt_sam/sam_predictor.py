# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch2trt import TRTModule
from typing import Tuple
import tensorrt as trt
import PIL.Image
import torch
import numpy as np
import torch.nn.functional as F
import time

import cv2
import matplotlib.pyplot as plt

from .owlvit_predictor import OwlVitPredictor
from .utils import calc_bounding

def load_mask_decoder_engine(path: str):
    
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    mask_decoder_trt = TRTModule(
        engine=engine,
        input_names=[
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input"
        ],
        output_names=[
            "iou_predictions",
            "low_res_masks"
        ]
    )

    return mask_decoder_trt


def load_image_encoder_engine(path: str):

    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, 'rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine,
        input_names=["image"],
        output_names=["image_embeddings"]
    )

    return image_encoder_trt


def preprocess_image(image, size: int = 512):

    if isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image)

    image_mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
    image_std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]

    image_pil = image
    aspect_ratio = image_pil.width / image_pil.height
    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    image_pil_resized = image_pil.resize((resize_width, resize_height))
    image_np_resized = np.asarray(image_pil_resized)
    image_torch_resized = torch.from_numpy(image_np_resized).permute(2, 0, 1)
    image_torch_resized_normalized = (image_torch_resized.float() - image_mean) / image_std
    image_tensor = torch.zeros((1, 3, size, size))
    image_tensor[0, :, :resize_height, :resize_width] = image_torch_resized_normalized

    return image_tensor.cuda()


def preprocess_points(points, image_size, size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points

def preprocess_mask(mask, image_size, size = 256):
    # Resize and pad mask
    height, width = image_size
    aspect_ratio = width / height

    if aspect_ratio >= 1:
        resize_width = size
        resize_height = int(size / aspect_ratio)
    else:
        resize_height = size
        resize_width = int(size * aspect_ratio)

    resized = cv2.resize(mask, (resize_width, resize_height))
    padded = np.zeros((256, 256))
    padded[0:resize_height, 0:resize_width] = resized
    
    # Set to floats and convert to tensor
    floated = np.array(padded, dtype=np.float32)
    tens = torch.torch.from_numpy(floated).cuda()

    return tens


def run_mask_decoder(mask_decoder_engine, features, points=None, point_labels=None, mask_input=None):
    if points is not None:
        assert point_labels is not None
        assert len(points) == len(point_labels)

    image_point_coords = torch.tensor([points]).float().cuda()
    image_point_labels = torch.tensor([point_labels]).float().cuda()

    if mask_input is None:
        mask_input = torch.zeros(1, 1, 256, 256).float().cuda()
        has_mask_input = torch.tensor([0]).float().cuda()
    else:
        has_mask_input = torch.tensor([1]).float().cuda()


    iou_predictions, low_res_masks = mask_decoder_engine(
        features,
        image_point_coords,
        image_point_labels,
        mask_input,
        has_mask_input
    )

    return iou_predictions, low_res_masks


def upscale_mask(mask, image_shape, size=256):
    
    if image_shape[1] > image_shape[0]:
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size

    mask[:, :, :lim_y, :lim_x]
    mask = F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape, mode='bilinear')
    
    return mask


class SAMPredictor(object):

    def __init__(self,
            image_encoder_engine: str,
            mask_decoder_engine: str,
            image_encoder_size: int = 1024,
            orig_image_encoder_size: int = 1024,
        ):
        self.image = None
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    def __predict(self, points, point_labels, mask_input=None, skip_upscale=False):
        if self.image is None:
            raise ValueError("You need to set an image to the predictor using set_image() first.")
        points = preprocess_points(
            points, 
            (self.image.height, self.image.width),
            self.orig_image_encoder_size
        )
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder_engine,
            self.features,
            points,
            point_labels,
            mask_input
        )

        if skip_upscale:
            return None, mask_iou, low_res_mask
        else:
            hi_res_mask = upscale_mask(
                low_res_mask, 
                (self.image.height, self.image.width)                           
            )
            return hi_res_mask, mask_iou, low_res_mask
    
    # Predict a mask, based on a list of points labeled as either background (0) or 
    # foreground (1). While you could pass in bounding box points labeled with 2 and 3,
    # why would you do that? Use the nice predict_bbox function I made for you. 
    # 
    # Inputs:
    #   - points: An ndarray of Nx2, where 2 values are x and y positions of points in the same coordinate frame as the image that was set.
    #   - point_labels: An array of Nx1, where the 1 value is a point label (0 for background, 1 for foreground)
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_points(self, foreground, background=None, iterations=1):
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        if foreground is None or len(foreground) < 1:
            raise ValueError("You must pass in foreground points.")

        if background is None or len(background) < 1:
            # Create labels of points
            fg_labs = [1] * len(foreground)

            # Cast to array if it hasn't already been done.
            points = np.array(foreground, dtype=np.float32)
            point_labels = np.array(fg_labs, dtype=np.float32)
        else:
            # Create labels of points
            fg_labs = [1] * len(foreground)
            bg_labs = [0] * len(background)

            # Cast to array if it hasn't already been done.
            points = np.array(np.concatenate((foreground, background)), dtype=np.float32)
            point_labels = np.array(np.concatenate((fg_labs, bg_labs)), dtype=np.float32)

        refine_mask = None
        for k in range(iterations):
            if (iterations - k) > 1:
                _, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask, skip_upscale=True)
                refine_mask = logits[0]
            else:
                mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)


        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        box = calc_bounding(mask)
        return [box], [mask]
            

    # Predict a mask, based on bounding box inputs. 
    # 
    # Inputs:
    #   - boxes: An 1x4 array, where the values of the rows are x1 y1 x2 y2, with (x1,y1) being the 
    #            top left point and (x2,y2) being the bottom right point. 
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_bbox(self, top_left, bot_right, iterations=1):
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        points = np.array([
            [top_left[0], top_left[1]],
            [bot_right[0], bot_right[1]]
        ])
        # Top left and bottom right points.
        point_labels = np.array([2, 3])
        
        refine_mask = None
        for k in range(iterations):
            if (iterations - k) > 1:
                _, iou, refine_mask = self.__predict(points, point_labels, mask_input=refine_mask, skip_upscale=True)
            else:
                mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)

        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        box = calc_bounding(mask)
        return [box], [mask]

    # Predict a mask, based on an input mask.
    # 
    # Inputs:
    #   - mask: A NxM ndarray, of the same size as the set image, with 0 representing noninclusion and 1 representing inclusion.
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_mask(self, mask, iterations=1):
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        # Select 10 random points of background
        points = np.array([[0,0]], dtype=np.float32)
        point_labels = np.array([0], dtype=np.float32)

        mask = preprocess_mask(mask,  (self.image.height, self.image.width))
        
        refine_mask = mask
        for k in range(iterations):
            if (iterations - k) > 1:
                _, iou, refine_mask = self.__predict(points, point_labels, mask_input=refine_mask, skip_upscale=True)
            else:
                mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)
                
        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        box = calc_bounding(mask)
        return [box], [mask]
    

    # Predict a mask, based on an input prompt. 
    # DO NOT USE THIS IF YOU WANT FINELY TAILORED OWL PROMPTING.
    # This is for quick, dirty, and easy prompt-driven SAM operations. If you want things like tree-organized prompts, 
    # image prompting, etc, use nano_owl and then bring the outputs from that into nano_sam.
    # 
    # Inputs:
    #   - prompt: A plaintext string of a prompt to OWL-Vit for 
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_prompt(self, prompt, iterations=1):
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        detector = OwlVitPredictor(0.1)
        detections = detector.predict(self.image, texts=prompt)

        masks = []
        boxes = []
        for d in detections:
            xmin, ymin, xmax, ymax = d['bbox']
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)

            boxes.append([xmin, ymin, xmax, ymax])
            _, m = self.predict_bbox([xmin, ymin], [xmax, ymax])
            masks.extend(m)

        return boxes, masks
        