import numpy as np
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