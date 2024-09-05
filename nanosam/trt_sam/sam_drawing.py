import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, overload
import numpy.typing as npt

def get_colors(count):
    cmap = plt.get_cmap("rainbow", count)
    colors = []
    for i in range(count):
        color = cmap(i)
        color = [int(255 * value) for value in color]
        colors.append(tuple(color[0:3]))
    return colors
    
@overload
def markup_image(
    image:npt.NDArray[np.float64], 
    masks:npt.NDArray[np.bool_],
    boxes:List[int], 
    ious:float,
    labels:Optional[str]
    ) -> npt.NDArray[np.float64]:
    # Definition of @overload function is not used. 
    ...

@overload
def markup_image(
    image:npt.NDArray[np.float64],
    masks:List[npt.NDArray[np.bool_]],
    boxes: List[List[str]],
    ious:List[float],
    labels:Optional[List[str]]
    )-> npt.NDArray[np.float64]:
    # Definition of @overload function is not used. 
    ...

'''
    Marks up an image using boxes, masks, and labels from SAM detections.

    Inputs:
        image: An OpenCV image (np.ndarray). 
        boxes: A bbox (list of [x0,y0,x1,y1]) or list therof.
        masks: A mask (np.ndarray of bools) or list therof.
        labels: A string or list therof.
    Outputs:
        marked image: The image with the input detections marked on it.
'''
def markup_image(
        image:npt.NDArray[np.float64],
        masks:npt.NDArray[np.bool_] | List[npt.NDArray[np.bool_]],
        boxes:List[int] | List[List[str]],
        ious:float | List[float],
        labels:Optional[str | List[str]]=None
        )-> npt.NDArray[np.float64]:
    
    # Allow input as singles or lists.
    if len(boxes) > 0  and not isinstance(boxes[0], list):
        boxes = [boxes]
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(ious, list):
        ious = [ious]
    if isinstance(labels, str):
        labels = [labels]


    if len(boxes) == 0 or len(masks) == 0 or len(labels) == 0:
        return image
    
    # Check some basic things about input data.
    if labels is None or len(labels) < 1:
        labels = ["object"] * len(masks)

    if len(labels) != len(boxes) != len(masks):
        raise ValueError(f"Lengths of lists of masks({len(masks)}), boxes({len(boxes)}), and labels({len(labels)}) do not match. ")

    running_mask = np.zeros(image.shape, image.dtype)
    running_bbox = np.zeros(image.shape, image.dtype)
    colors = get_colors(len(masks))

    for k, m, b, i, l in sorted(zip(range(len(masks)), masks, boxes, ious, labels), key=lambda set:set[3]):
        c = colors[k]
        running_mask = draw_mask(running_mask, m, c, l)
        running_bbox = draw_box(running_bbox, b, c, i, l)

    image = cv2.addWeighted(running_mask, 0.85, image, 1, 0, image)
    coords = np.nonzero(running_bbox)
    image[coords[0], coords[1],:] = 0
    image[coords] = running_bbox[coords]

    return image


def draw_mask(img, mask, color, label=None):
    m = mask.astype(np.uint8)
    colorImg = np.zeros(img.shape, img.dtype)
    colorImg[:,:] = color
    colorMask = cv2.bitwise_and(colorImg, colorImg, mask=m)
    img = cv2.bitwise_or(img, colorMask)
    return img

def draw_box(image, box, color, iou, label=None):
    if len(box) < 1:
        return image
    xmin, ymin, xmax, ymax = box
    
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 3)
    image = cv2.rectangle(image, (xmin-2, ymin-40), (xmin+300, ymin), color, -1)
    cv2.putText(image, f"{iou:.2f} |", (xmin+10, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),2)
    cv2.putText(image, label, (xmin+115, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),2)
    
    return image