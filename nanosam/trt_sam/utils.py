import cv2
import numpy as np
from matplotlib import colormaps

def calc_bounding(mask):
    x,y = np.where(mask == True)
    return [np.min(x), np.min(y), np.max(x), np.max(y)]

def random_color():
    c = np.random.choice(range(256), size=3)
    return (int(c[0]), int(c[1]), int(c[2]))

def mpl_color(k):
    cmap = [[200,0,0], [0,200,0], [0,0,200], [100,100,0], [100,0,100], 
            [0,100,100], [100,100,100], [50,50,0], [0,50,50], [50,0,50], 
            [50,50,50]]
    if k > len(cmap):
        return random_color
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