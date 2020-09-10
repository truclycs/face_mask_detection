import numpy as np


def single_class_non_max_suppression(bboxes, 
                                        confidences, 
                                        conf_thresh=0.2, 
                                        iou_thresh=0.5):
    """Do nms on single class.
    Hint: For the specific class, given the bbox and its confidence,
    1) Sort the bbox according to the confidence from top to down, 
    we call this a set.
    2) Select the bbox with the highest confidence, remove it from set, 
    and do IOU calculate with the rest bbox.
    3) Remove the bbox whose IoU is higher than the iou_thresh from the set.
    4) Loop step 2 and 3, until the set is empty.

    Args:
        bboxes (ndarray): numpy array of 2D, [num_bboxes, 4].
        confidences (ndarray): numpy array of 1D, [num_bboxes].
        conf_thresh (float): confidence score.
        iou_thresh (float): IoU threshold of NMS.

    Returns:
        result (list): A list of filtered proposals.

    """

    if len(bboxes) == 0: 
        return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs):
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    return conf_keep_idx[pick]
