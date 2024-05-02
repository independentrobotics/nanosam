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
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    def __predict(self, points, point_labels, mask_input=None):
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
    #   - iou_thresh: The IOU threshold to cut off mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_points(self, points, point_labels, iterations=1, iou_thresh=0.5):
        if iterations != 1 or iou_thresh != 0.5:
            raise NotImplementedError("Sorry, multiple iterations of prediction are not yet supported.")
        
        refine_mask = None
        for k in range(iterations):
            mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)

        return mask
            

    # Predict a mask, based on bounding box inputs. 
    # 
    # Inputs:
    #   - boxes: An 1x4 array, where the values of the rows are x1 y1 x2 y2, with (x1,y1) being the 
    #            top left point and (x2,y2) being the bottom right point. 
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    #   - iou_thresh: The IOU threshold to cut off mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_bbox(self, bbox, iterations=1, iou_thresh=0.5):
        if iterations != 1 or iou_thresh != 0.5:
            raise NotImplementedError("Sorry, multiple iterations of prediction are not yet supported.")
        
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        # Top left and bottom right points.
        point_labels = np.array([2, 3])
        
        refine_mask = None
        for k in range(iterations):
            mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)

        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        return mask

    # Predict a mask, based on an input mask.
    # 
    # Inputs:
    #   - mask: A NxM ndarray, of the same size as the set image, with 0 representing noninclusion and 1 representing inclusion.
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    #   - iou_thresh: The IOU threshold to cut off mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_mask(self, mask, iterations=1, iou_thresh=0.5):
        if iterations != 1 or iou_thresh != 0.5:
            raise NotImplementedError("Sorry, multiple iterations of prediction are not yet supported.")
        
        refine_mask = None
        for k in range(iterations):
            mask, iou, logits = self.__predict(points, point_labels, mask_input=refine_mask)

    # Predict a mask, based on an input prompt. 
    # DO NOT USE THIS IF YOU WANT FINELY TAILORED OWL PROMPTING.
    # This is for quick, dirty, and easy prompt-driven SAM operations. If you want things like tree-organized prompts, 
    # image prompting, etc, use nano_owl and then bring the outputs from that into nano_sam.
    # 
    # Inputs:
    #   - prompt: A plaintext string of a prompt to OWL-Vit for 
    #   - iterations: The number of iterations to attempt mask refinement, if desired. 
    #   - iou_thresh: The IOU threshold to cut off mask refinement, if desired. 
    # Outputs:
    #   - Mask (high resolution mask from internal __predict function)
    # 
    def predict_prompt(self, prompt, iteractions=1, iou_thresh=0.5):
        raise NotImplementedError("Internal OWL prompting is not yet implemented, sorry.")