# This file provides simple setup utility functions for the OwlPredictor integrated into
# nanosam. A reminder, if you haven't seen this elsewhere: if you want to do anything even
# remotely complex, you should be using nanoowl instead of the integrated predictor.
from ir_utils.filesystem_tools import get_dl_model_directory, create_dl_model_directory
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection
)
import os

def setup():
    download_owl_model()

def download_owl_model():
    try:
        model_path = get_dl_model_directory("nanosam-int-owl")
    except ValueError:
        create_dl_model_directory("nanosam-int-owl")
 
    OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir=model_path)
    OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir=model_path)


if __name__ == "__main__":
    setup()