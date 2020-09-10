import numpy as np


def generate_anchors(feature_map_sizes, 
                        anchor_sizes, 
                        anchor_ratios, 
                        offset=0.5):
    """Generate anchors.

    Args:
        feature_map_sizes (2D list): list of list, for example: [[40,40], [20,20]]
        anchor_sizes (2D list): list of list, for example: [[0.05, 0.075], [0.1, 0.15]]
        anchor_ratios (2D list): list of list, for example: [[1, 0.5], [1, 0.5]]
        offset (float): default to 0.5

    Returns:
        result (ndarray): anchors

    """
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)

        num_anchors = len(anchor_sizes[idx]) +  len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []

        # different scales with the first aspect ratio
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0] # select the first ratio
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        # the first scale, with different aspect ratios (except the first one)
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0] # select the first scale
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])

        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)

    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


def decode_bbox(anchors, 
                raw_outputs, 
                variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode the actual bbox according to the anchors.
    The anchor value order is: [xmin,ymin, xmax, ymax]
    
    Args:
        anchors (ndarray): numpy array with shape [batch, num_anchors, 4]
        raw_outputs (ndarray): numpy array with the same shape with anchors
        variances (list): list of float, default=[0.1, 0.1, 0.2, 0.2]
    
    Returns:
        result (ndarray): anchors.

    """    
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]

    raw_outputs_rescale = raw_outputs * np.array(variances)
    
    predict_center_x = raw_outputs_rescale[:, :, 0:1]*anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2]*anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3])*anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:])*anchors_h
    
    predict_xmin = predict_center_x - predict_w/2
    predict_ymin = predict_center_y - predict_h/2
    predict_xmax = predict_center_x + predict_w/2
    predict_ymax = predict_center_y + predict_h/2
    
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)

    return predict_bbox