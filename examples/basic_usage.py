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
import PIL.Image
import argparse, sys

from nanosam.trt_sam import SAMPredictor, markup_image

asset_path = "/opt/nanosam/assets/"
outpath = "/root/out/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--points',action='store_true', help="Test point-based input.")
    parser.add_argument('-b', '--bbox',action='store_true', help="Test bounding box input.")
    parser.add_argument('-m', '--mask',action='store_true', help="Test mask input.")
    parser.add_argument('-o', '--owl',action='store_true', help="Test OWL-ViT prompt input.")
    parser.add_argument('-a', '--amg',action='store_true', help="Test automatic mask generation.")
    parser.add_argument('-s', '--sev',action='store_true', help="Test several masks being plotted.")
    
    args = parser.parse_args()

    if not (args.points or args.bbox or args.mask or args.owl or args.amg or args.sev or args.all):
        print("NO INPUT TYPE SPECIFIED. Use one of the options listed below. ")
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # Instantiate TensorRT predictor
    predictor = SAMPredictor()

    # Read image and run image encoder
    image = cv2.imread( asset_path + "dogs.jpg")
    predictor.set_image(image)

    if args.points:
        # # Segment using points.

        fg_points = [[225, 225], [400,400], [650,350], [350,650]]
        bg_points = [[700, 150]]
        
        mask, box, iou = predictor.predict_points(fg_points, bg_points, iterations=3)

        img = cv2.imread(asset_path + "dogs.jpg")
        out = markup_image(img, mask, box, iou)
        cv2.imwrite(outpath+"sam_points.jpg", out)

    if args.bbox:
        # Segment using bounding box
        # x0, y0, x1, y1
        bbox = [100,100, 850,759]

        mask, box, iou = predictor.predict_bbox(bbox, iterations=3)
        img = cv2.imread(asset_path + "dogs.jpg")
        out = markup_image(img, mask, box, iou)
        cv2.imwrite(outpath+"sam_bbox.jpg", out)
                    
    if args.mask:
        # Segment using mask (here provided by an opencv color threshold.)
        img = cv2.imread(asset_path + "dogs.jpg")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([3, 0, 120])
        upper_range = np.array([15, 255, 255])
        thresh = cv2.inRange(hsv, lower_range, upper_range)

        mask, box, iou = predictor.predict_mask(thresh, iterations=3)
        out = markup_image(img, mask, box, iou)
        cv2.imwrite(outpath+"sam_mask.jpg", out)

    if args.owl:
        img = cv2.imread(asset_path + "dogs.jpg")
        prompt = "a dog"
        
        mask, box, iou = predictor.predict_prompt(prompt, iterations=3)
        
        out = markup_image(img, mask, box, iou, prompt)
        cv2.imwrite(outpath+"sam_prompt.jpg", out)
    
    if args.sev:
        img = cv2.imread(asset_path + "dogs.jpg")

        fg_points = [[225, 225], [400,400], [650,350], [350,650]]
        bg_points = [[700, 150]]
        mask1, box1, iou1 = predictor.predict_points(fg_points, bg_points, iterations=3)

        fg_points = [[800,225], [850, 450], [790,600]]
        bg_points = [[700, 150]]
        mask2, box2, iou2 = predictor.predict_points(fg_points, bg_points, iterations=3)

        masks = [mask1, mask2]
        boxes = [box1, box2]
        ious = [iou1, iou2]
        
        out = markup_image(img, masks, boxes, ious)
        cv2.imwrite(outpath+"sam_several_masks.jpg", out)


    if args.amg:
        img = cv2.imread(asset_path + "dogs.jpg")
        masks, boxes, ious = predictor.automatic_mask_generation(img)
        
        out = markup_image(img, masks, boxes, ious)
        cv2.imwrite(outpath+"sam_amg.jpg", out)

 