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
import PIL.Image
import argparse

from nanosam.trt_sam.sam_predictor import SAMPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="../data/resnet18_image_encoder.engine")
    parser.add_argument("--mask_decoder", type=str, default="../data/mobile_sam_mask_decoder.engine")
    parser.add_argument('-p', '--points',action='store_true')
    parser.add_argument('-b', '--bbox',action='store_true')
    parser.add_argument('-m', '--mask',action='store_true')
    parser.add_argument('-s', '--prompt',action='store_true')
    
    args = parser.parse_args()
        
    # Instantiate TensorRT predictor
    predictor = SAMPredictor(
        args.image_encoder,
        args.mask_decoder
    )

    # Read image and run image encoder
    image = PIL.Image.open("../assets/dogs.jpg")
    predictor.set_image(image)

    if args.points:
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

    if args.bbox:
        # Segment using bounding box
        tl = [100,100]  # x0, y0, x1, y1
        br = [850,759]

        mask = predictor.predict_bbox(tl, br)

        # Draw results
        plt.clf()
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)

        x = [tl[0], br[0], br[0], tl[0], tl[0]]
        y = [tl[1], tl[1], br[1], br[1], tl[1]]
        plt.plot(x, y, 'g-')
        plt.savefig("/out/basic_usage_bbox_out.jpg")

    if args.mask:
        # Segment using mask (here provided by an opencv color threshold.)
        img = cv2.imread('../assets/dogs.jpg')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([3, 0, 120])
        upper_range = np.array([15, 255, 255])
        thresh = cv2.inRange(hsv, lower_range, upper_range)
        mask = predictor.predict_mask(thresh, iterations=2)

        plt.clf()
        
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5)
        plt.savefig("/out/basic_usage_mask_out.jpg")