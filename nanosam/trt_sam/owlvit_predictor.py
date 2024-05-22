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

import requests
import PIL.Image
import torch
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection
)
from typing import Sequence, List, Tuple
from .build_owl_model import download_owl_model
from ir_utils.filesystem_tools import get_dl_model_directory

class OwlVitPredictor(object):
    def __init__(self, threshold=0.1):
        download_owl_model()
        model_path = get_dl_model_directory("nanosam-int-owl")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir=model_path)
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir=model_path)
        self.threshold = threshold

    def predict(self, image: PIL.Image.Image, texts: Sequence[str]):
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=self.threshold)
        i = 0
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            detection = {"bbox": box.tolist(), "score": float(score), "label": int(label), "text": texts[label]}
            detections.append(detection)
        return detections

