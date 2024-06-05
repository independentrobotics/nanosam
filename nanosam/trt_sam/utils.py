import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, overload
import numpy.typing as npt
from scipy.cluster.vq import kmeans

def calc_bounding(mask, padding=3):
    x = np.where(np.any(mask == True, axis=0))[0]
    y = np.where(np.any(mask == True, axis=1))[0]
    if x.size == 0 or y.size == 0:
        return []

    
    h,w = mask.shape
    _, xmin, _ = sorted([0, np.min(x)-2, w])
    _, ymin, _ = sorted([0, np.min(y)-2, h])
    _, xmax, _ = sorted([0, np.max(x)+2, w])
    _, ymax, _ = sorted([0, np.max(y)+2, h])
    
    return [xmin, ymin, xmax, ymax]

def box_area(box):
    w = box[2] - box[0]
    l = box[3] - box[1]
    return w * l

def box_contained(meta, box):
    mx1, my1, mx2, my2 = meta
    bx1, by1, bx2, by2 = box
    return (bx1 > mx1) and ( by1 > my1) and (bx2 < mx2) and (by2 < my2)

def calc_mask_candidates(mask, n_candidates=10):
    obs = np.transpose(np.nonzero(mask)).astype(np.float32)
    candidates, _ = kmeans(obs, n_candidates, iter=5)

    return candidates.astype(np.int32)


def prune_owl_detections(detections, max_detections=5):
    # Filter by max scores
    if len(detections) < max_detections:
        filtered = detections
    else: 
        scores = np.array([d['score'] for d in detections])
        ind = np.argpartition(scores, -max_detections)[-max_detections:]
        filtered = [detections[i] for i in ind]

    # if, after max score filtering, we only have 1 detection left, no need for
    # area-based filtering.
    if len(filtered) > 1:
        # Filter out the largest bounding box if it  fully contain all others.
        areas = [box_area(d['bbox']) for d in filtered]
        
        # KLUDGE 
        if (len(areas) == 0):
            return detections

        ind_max = areas.index(max(areas))

        check = True 
        max_box = filtered[ind_max]
        for k, d in enumerate(filtered):
            if k == ind_max:
                continue
            
            check = check and box_contained(max_box['bbox'], d['bbox'])

        if check:
            del filtered[ind_max]

    return filtered

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
    image:npt.NDArray[np.float_], 
    masks:npt.NDArray[np.bool_],
    boxes:List[int], 
    ious:float,
    labels:Optional[str]
    ) -> npt.NDArray[np.float_]:
    # Definition of @overload function is not used. 
    ...

@overload
def markup_image(
    image:npt.NDArray[np.float_],
    masks:List[npt.NDArray[np.bool_]],
    boxes: List[List[str]],
    ious:List[float],
    labels:Optional[List[str]]
    )-> npt.NDArray[np.float_]:
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
        image:npt.NDArray[np.float_],
        masks:npt.NDArray[np.bool_] | List[npt.NDArray[np.bool_]],
        boxes:List[int] | List[List[str]],
        ious:float | List[float],
        labels:Optional[str | List[str]]=None
        )-> npt.NDArray[np.float_]:
    
    # Allow input as singles or lists.
    if len(boxes) < 1 or not isinstance(boxes[0], list):
        boxes = [boxes]
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(ious, list):
        ious = [ious]
    if isinstance(labels, str):
        labels = [labels]
    
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