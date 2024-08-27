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

import PIL.Image
import numpy as np
import torch
import tensorrt as trt
from torch2trt import TRTModule
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from math import floor

from typing import Optional, Union, Tuple, List, overload
import numpy.typing as npt

from .build_trt_engines import build_engines
from .owlvit_predictor import OwlVitPredictor
from .utils import calc_bounding, calc_mask_candidates, prune_owl_detections

from ir_utils.filesystem_tools import get_dl_model_directory

'''
    Internal use only, this function loads the mask decoder TRT engine. 
'''
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


'''
    Internal use only, this function loads the image encoder TRT engine. 
'''
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


'''
    Internal use only, this function preprocesses an image for use in SAM.
'''
def preprocess_image(image: PIL.Image.Image, size: int = 512):

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

'''
    Internal use only, this function performs point preprocessing. 
'''
def preprocess_points(points: npt.NDArray[np.float64], image_size: Tuple[int, int], size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points

'''
    Internal use only, this function performs mask preprocessing.
'''
def preprocess_mask(mask: npt.NDArray[np.float64], image_size: Tuple[int, int], size = 256):

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
    padded = np.zeros((1,1,256, 256))
    padded[:, :, 0:resize_height, 0:resize_width] = resized
    
    # Set to floats and convert values to more SAM-like logits values (guessing)
    floated = np.array(padded, dtype=np.float32)
    floated *= (16.0/floated.max())
    floated[floated==0] = -16.0
    
    tens = torch.torch.from_numpy(floated).cuda()

    return tens


'''
    Internal use only, this function performs the actual operation of mask decoding using a TRT engine.
'''
def run_mask_decoder(mask_decoder_engine: TRTModule, features, points:npt.NDArray[np.float64]=None, point_labels:npt.NDArray[np.float64]=None, mask_input:npt.NDArray[np.float64]=None):
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

'''
    Internal use only, this function upscales a mask from its original size to the image
    size.
'''
def upscale_mask(mask:npt.NDArray[np.float64], image_shape:Tuple[int, int], size:int=256):
    
    if image_shape[1] > image_shape[0]:
        lim_x = size
        lim_y = int(size * image_shape[0] / image_shape[1])
    else:
        lim_x = int(size * image_shape[1] / image_shape[0])
        lim_y = size

    mask[:, :, :lim_y, :lim_x]
    mask = F.interpolate(mask[:, :, :lim_y, :lim_x], image_shape, mode='bilinear')
    
    return mask


'''
    SAMPredictor
'''
class SAMPredictor(object):

    def __init__(self,
            image_encoder_size: int = 1024,
            orig_image_encoder_size: int = 1024,
        ):

        # This function checks to see if TRT engines have already been built and stored in the 
        # appropriate IR deep learning models folder. If not, it builds them.
        build_engines()        

        model_path = get_dl_model_directory("nanosam")
        image_encoder_engine = model_path + 'resnet18_image_encoder.engine'
        mask_decoder_engine = model_path + 'mobile_sam_mask_decoder.engine'

        self.image = None
        self.image_encoder_engine = load_image_encoder_engine(image_encoder_engine)
        self.mask_decoder_engine = load_mask_decoder_engine(mask_decoder_engine)
        self.image_encoder_size = image_encoder_size
        self.orig_image_encoder_size = orig_image_encoder_size

    '''
        set_image
    '''
    def set_image(self, image: PIL.Image.Image | npt.NDArray):
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        elif not isinstance(image, PIL.Image.Image):
            raise TypeError(f"To set image, convert it to a PIL.Image or a numpy.ndarray (OpenCV image). Type {type(image)} is not supported")

        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder_engine(self.image_tensor)

    '''
       The actual prediction function that produces masks. This function is not intended to be called
       publicly, because the interface provided to users is more flexible and comprehensive. 

        Inputs:
            points: An ND array of size 2xN (where N < 10), providing the input prompts for SAM.
            point_labels: An ND array of size 1xN, providing labels for the input points. 
            mask_input: An optional input, an ND array of the same size as the input image.
            iterations: The number of times to re-predict the mask to refine the prediction.

        Outputs: 
            mask: A boolean ND array of image size.
            box: A bounding box, passed as [xmin, ymin, xmax, ymax]
            iou: A floating point number indicating the quality of the mask.
    '''
    def __predict(
            self, 
            points:npt.NDArray[np.float64], 
            point_labels:npt.NDArray[np.float64], 
            mask_input:Optional[npt.NDArray[np.float64]]=None, 
            iterations:int=1
            ):
        
        if self.image is None:
            raise ValueError("You need to set an image to the predictor using set_image() first.")
        
        points = preprocess_points(
            points, 
            (self.image.height, self.image.width),
            self.orig_image_encoder_size
        )

        # Now for the actual prediction step.
        last_mask = None
        last_score = 0
    
        # TODO Consider offering all four masks as a return from the model.
        if iterations > 1:
            # We only need to do this if the number of iterations is greater than 1.
            for k in range(iterations):
                iou_preds, logits = run_mask_decoder(self.mask_decoder_engine,self.features,points,point_labels,mask_input)

                # Get the maximum IOU
                max_idx = iou_preds.argmax()
                iou = iou_preds[0, max_idx].detach().cpu().numpy()

                # If this last iteration helped, then let's continue. 
                if (iterations - k) > 1 and iou > last_score:
                    last_score = iou
                    mask_input = logits[:, max_idx-1:max_idx, :, :]
                # Otherwise, upscale the mask and break out of the loop.
                else:
                    break
            
                last_mask = logits

            hi_res = upscale_mask(last_mask, (self.image.height, self.image.width))
            mask = (hi_res[0, max_idx] > 0).detach().cpu().numpy()

        # If we only do one iteration, we have to take the mask as is. 
        else:
            iou_preds, logits = run_mask_decoder(self.mask_decoder_engine,self.features,points,point_labels,mask_input)
            hi_res = upscale_mask(logits, (self.image.height, self.image.width))

            max_idx = iou_preds.argmax()
            mask = (hi_res[0, max_idx] > 0).detach().cpu().numpy() 
            iou = iou_preds[0, max_idx].detach().cpu().numpy()

        bbox = calc_bounding(mask)
        iou = float(iou)
        return mask, bbox, iou
        
    '''
        Predict a mask, based on lists of foreground and background points (MAX 10 points total).
        
        Inputs:
            foregound: A list of points (each point is a list of two ints (x,y)) that are believed to be part of an object. 
            background: A list of points (each point is is a list of two ints (x,y) that are believed to be background.
            iterations: The number of times to re-process the mask. If an iteration fails to produce a higher scoring mask, the previous mask will be returned. 
        Outputs:
            Mask (high resolution mask)
            Bounding Box ([xmin, ymin, xmax, ymax])
            IOU (a float indicating the quality of the mask.)
    '''
    def predict_points(self, foreground:List[List[int]], background:List[List[int]]=None, iterations:int=1) -> Tuple[npt.NDArray[np.bool_], List[int], float]:
        # Input checking.
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        if foreground is None or len(foreground) < 1:
            raise ValueError("You must pass foreground points.")
        
        if (len(foreground) + len(background)) > 10:
            raise ValueError(f"SAM only accepts <10 points as input, reduce the number of points in foreground (N:{len(foreground)}) and background (N:{len(background)}) lists.")

        # Now that inputs have been confirmed, we can get everything in an appropriate shape.
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

        return self.__predict(points, point_labels, iterations=iterations)

    '''
        Predict a mask, based on bounding box input. 
        
        Inputs:
            bbox: A bounding box, organized as [xmin, ymin, xmax, ymax]
            iterations: The number of times to re-process the mask. If an iteration fails to produce a higher scoring mask, the previous mask will be returned. 
        Outputs:
            Mask (high resolution mask)
            Bounding Box ([xmin, ymin, xmax, ymax])
            IOU (a float indicating the quality of the mask.)
    '''
    def predict_bbox(self, bbox: List[int], iterations:int=1) -> Tuple[npt.NDArray[np.bool_], List[int], float]:
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        # Top left and bottom right points.
        point_labels = np.array([2, 3])
        
        return self.__predict(points, point_labels, iterations=iterations)

    '''
        Predict a mask, based on a mask input. 
        
        Inputs:
            mask: An NDArray of the same size as the image, floating point. 
            iterations: The number of times to re-process the mask. If an iteration fails to produce a higher scoring mask, the previous mask will be returned. 
        Outputs:
            Mask (high resolution mask)
            Bounding Box ([xmin, ymin, xmax, ymax])
            IOU (a float indicating the quality of the mask.)
    '''
    def predict_mask(self, mask:npt.NDArray[np.float64], iterations:int=1) -> Tuple[npt.NDArray[np.bool_], List[int], float]:
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        centroids = calc_mask_candidates(mask)
        
        # Select 10 points inside positive mask
        points = np.array([[p[0], p[1]] for p in centroids], dtype=np.float32)
        point_labels = np.array([1]*centroids.shape[0], dtype=np.float32)

        mask = preprocess_mask(mask,  (self.image.height, self.image.width))

        return self.__predict(points, point_labels, mask_input= mask, iterations=iterations)

    '''
        Predict a mask, based on a natural language input.  
        
        Inputs:
            prompt: A string containing a natural language input.
            iterations: The number of times to re-process the mask. If an iteration fails to produce a higher scoring mask, the previous mask will be returned. 
        Outputs:
            Mask (high resolution mask)
            Bounding Box ([xmin, ymin, xmax, ymax])
            IOU (a float indicating the quality of the mask.)
    '''
    def predict_prompt(self, prompt:str, iterations:int=1) -> Tuple[npt.NDArray[np.bool_], List[int], float]:
        if iterations < 1:
            raise ValueError(f"Iteractions cannot be less than 1, you passed iterations={iterations}")
        
        detector = OwlVitPredictor(0.1)
        detections = detector.predict(self.image, texts=[prompt])
        d = prune_owl_detections(detections, max_detections=1)[0]
        xmin, ymin, xmax, ymax = d['bbox']
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        return self.predict_bbox([xmin, ymin, xmax, ymax], iterations=iterations)
        

    def automatic_mask_generation(self, image):
        raise NotImplementedError("Automatic mask generation is not implemented.")