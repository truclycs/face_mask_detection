import __init__
from pathlib import Path
import json
import numpy as np
import cv2
import torch
from ailibs.__init__ import timeit

from ailibs.dectector.ssd.utils.anchor import generate_anchors, decode_bbox
from ailibs.dectector.ssd.utils.nms import single_class_non_max_suppression


class FaceMaskDetector:
    """ Face mask detector class """    
    def __init__(self, 
                    model_path: Path=None,
                    conf_thresh=0.75,
                    iou_thresh=0.4,
                    target_shape=(360, 360),
                    **kwargs):
        """ Constructor.
        
        Args:
            conf_thresh (float): the min threshold of classification probabity.
            iou_thresh (float): the IOU threshold of NMS.
            target_shape (tuple): the model input size.

        """
        self.model_path = model_path
        self.__model = torch.load(self.model_path)

        feature_map_sizes = [[45, 45], 
                                [23, 23], 
                                [12, 12], 
                                [6, 6], 
                                [4, 4]]

        anchor_sizes = [[0.04, 0.056], 
                        [0.08, 0.11], 
                        [0.16, 0.22], 
                        [0.32, 0.45], 
                        [0.64, 0.72]]

        anchor_ratios = [[1, 0.62, 0.42]] * 5

        anchors = generate_anchors(feature_map_sizes, 
                                        anchor_sizes, 
                                        anchor_ratios)
    
        self.anchors_exp = np.expand_dims(anchors, axis=0)    

        #Pick GPU if available, else CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.__model.to(self.device)        

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.target_shape = target_shape
        self.log = kwargs.get('log', False)

    def detect_noses(self, image):
        """ Detect nose in image.

        Args:
            image (ndarray): 3D numpy array of image.

        Returns:
            result (int): 1 if has nose else 0.

        """
        ds_factor = 0.5
        frame = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 5)
        return 1 if len(nose_rects) else 0
    
    @timeit
    def detect(self, 
                image,
                detect_nose=True):
        """ Detect face mask in image.

        Args:
            image (ndarray): 3D numpy array of image.
            detect_nose (bool): detect nose or not.
        
        Returns:
            result (dict): Faces information.

        """
        faces_info = []

        # Preprocess
        height, width, _ = image.shape
        image_resized = cv2.resize(image, self.target_shape)
        image_np = image_resized / 255
        image_exp = np.expand_dims(image_np, axis=0)
        image_transposed = image_exp.transpose((0, 3, 1, 2))

        # Detect
        input_tensor = torch.tensor(image_transposed).float().to(self.device)
        y_bbox, y_score, = self.__model.forward(input_tensor)
        y_bboxes_output, y_cls_output = y_bbox.detach().cpu().numpy(), y_score.detach().cpu().numpy()

        # Remove the batch dimension, for batch is always 1
        y_bboxes = decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
        # keep_idx is the alive bounding box after NMS
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                        bbox_max_scores,
                                                        conf_thresh=self.conf_thresh,
                                                        iou_thresh=self.iou_thresh,
                                                        )
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # Clip the coordinate, avoid the value exceed the image boundary
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            # Detect nose
            if detect_nose and class_id == 0:
                class_id = self.detect_noses(image)

            faces_info.append([int(class_id), 
                                float(conf), 
                                int(xmin), 
                                int(ymin), 
                                int(xmax), 
                                int(ymax)])
                                
        list_info = {}
        list_info['info'] = json.dumps(faces_info)              
        return list_info