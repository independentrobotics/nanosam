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

import numpy as np
import cv2 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image
import argparse

from nanosam.trt_sam.sam_predictor import SAMPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="../data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="../data/mobile_sam_mask_decoder.engine")
    args = parser.parse_args()
        
    # Instantiate TensorRT predictor
    predictor = SAMPredictor(
        args.image_encoder,
        args.mask_decoder
    )

    # Read image and run image encoder
    image = PIL.Image.open("../assets/dogs.jpg")
    predictor.set_image(image)

    # # Segment using points.

    fg_points = [[225, 225], [400,400], [650,350], [350,650]]
    bg_points = [[700, 150]]
    
    mask = predictor.predict_points(fg_points, bg_points, iterations=2)

    # Draw results
    plt.clf()
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)

    # I'm just doing this for easier slicing
    fg_points = np.array(fg_points)
    bg_points = np.array(bg_points)

    x = fg_points[:, 0]
    y = fg_points[:, 1]
    plt.plot(x, y, 'bo')

    x = bg_points[:, 0]
    y = bg_points[:, 1]
    plt.plot(x, y, 'ro')

    plt.savefig(f"/out/basic_usage_points_out.jpg")

    # Segment using bounding box
    tl = [850, 759]  # x0, y0, x1, y1
    br = [850,759]

    mask = predictor.predict_bbox(tl, br)

    # Draw results
    plt.clf()
    
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)

    # HACK So this is really dumb, but for some reason I can't figure out
    # the version I wrote to draw the rectangle from th tl/br variables just...
    # doesn't show up on the saved image? Not sure why. 
    bbox = [100, 100, 850, 759]  # x0, y0, x1, y1
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, 'y-')
    plt.savefig("/out/basic_usage_bbox_out.jpg")

    # Segment using mask (here provided by an opencv color threshold.)


