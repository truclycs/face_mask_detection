import __init__
import cv2
from base_camera import BaseCamera
import requests
import base64
import json
import os
import time

from streaming.read_info import camera_source, api
from bgsubtraction.BGSubtractor import BGSubtractor, BGSubtractionParammeter
min_area = BGSubtractionParammeter.minArea
threshold = BGSubtractionParammeter.threshold
padding = BGSubtractionParammeter.padding
detector_bgs = BGSubtractor(threshold, min_area, padding)


class Camera(BaseCamera):
    video_source = camera_source

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        # from run import api
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        frame_count = 0
        id2class = {0: 'mask', 
                    1: 'no_mask'}
        scalefactor = 0.75
        
        status = True
        while status:
            # read current frame
            status, img = camera.read()
            if not status:
                break

            frame_count += 1
            if frame_count % 5 != 0:
                # img1 = cv2.flip(img, 0)
                # out.write(img1)
                yield cv2.imencode('.jpg', img)[1].tobytes()
                continue

            image = cv2.resize(img, (0,0), fx=scalefactor, fy=scalefactor)
            movingObjects = detector_bgs.detect(img)
            
            print("mmm", frame_count, movingObjects)
            if movingObjects:
                print("detect")
                _, buff = cv2.imencode('.jpg', img)
                jpg_as_text = base64.b64encode(buff)

                response = requests.post(api, json={'img': jpg_as_text}).json()
                
                face_info = json.loads(response['info'])
                # Draw bounding box
                for (class_id, conf, xmin, ymin, xmax, ymax) in face_info:
                    if class_id == 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(img, 
                                "%s: %.2f" % (id2class[class_id], conf), 
                                (xmin + 2, ymin - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, 
                                color)

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
        
        # # cap.release()
        # print('here')
        # out.release()
