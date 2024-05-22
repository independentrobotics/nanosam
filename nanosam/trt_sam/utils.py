import cv2
import numpy as np
from matplotlib import colormaps

def calc_bounding(mask):
    x,y = np.where(mask == True)
    print(x,y)
    if len(x) < 1 or len(y) < 1:
        return []
        
    return [np.min(x), np.min(y), np.max(x), np.max(y)]

def box_area(box):
    w = box[2] - box[0]
    l = box[3] - box[1]
    return w * l

def box_contained(meta, box):
    mx1, my1, mx2, my2 = meta
    bx1, by1, bx2, by2 = box
    return (bx1 > mx1) and ( by1 > my1) and (bx2 < mx2) and (by2 < my2)


def prune_owl_detections(detections, max_detections=5):
    # Filter by max scores
    if len(detections) < max_detections:
        filtered = detections
    else: 
        scores = np.array([d['score'] for d in detections])
        ind = np.argpartition(scores, -max_detections)[-max_detections:]
        filtered = [detections[i] for i in ind]

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


def random_color():
    c = np.random.choice(range(256), size=3)
    return (int(c[0]), int(c[1]), int(c[2]))

def mpl_color(k):
    cmap = [[200,0,0], [0,200,0], [0,0,200], [100,100,0], [100,0,100], 
            [0,100,100], [100,100,100], [50,50,0], [0,50,50], [50,0,50], 
            [50,50,50]]
    if k >= len(cmap):
        return random_color()
    else:
        return cmap[k]

def markup_image(image, labels, boxes, masks):
    if labels is None or len(labels) < 1:
        labels = [""] * len(masks)

    if len(labels) != len(boxes) != len(masks):
        raise ValueError(f"Lengths of lists of masks({len(masks)}), boxes({len(boxes)}), and labels({len(labels)}) do not match. ")

    running = image
    for k, l, b, m in zip(range(len(masks)), labels, boxes, masks):
        c = mpl_color(k)
        running = draw_mask(running, m, c, l)
        running = draw_box(running, b, c, l)

    return running


def draw_mask(img, mask, color, label=None):
    m = mask.astype(np.uint8)
    colorImg = np.zeros(img.shape, img.dtype)
    colorImg[:,:] = color
    colorMask = cv2.bitwise_and(colorImg, colorImg, mask=m)

    image = cv2.addWeighted(colorMask, 0.5, img, 1, 0, img)
    return img

def draw_box(image, box, color, label=None):
    xmin, ymin, xmax, ymax = box
    
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(image, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color,2)
    return image