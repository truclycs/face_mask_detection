import __init__
import cv2
from base_camera import BaseCamera
import requests
import base64
import RPi.GPIO as GPIO
from time import sleep
import json 
import time
import os
import dlib

from streaming.read_info import camera_source, api
from ailibs.tracker.FaceTracker import FaceTracker

PORT_PI = 8
ALERT = 4
FACE_TRACKERS = FaceTracker(log=True)


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

        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
        # Set pin 8 to be an output pin and set initial value to low (off)
        GPIO.setup(PORT_PI, GPIO.OUT, initial=GPIO.LOW) 

        id2class = {0: 'Mask', 
                    1: 'NoMask'}

        frame_count = 0    
        alerting = False 
        count_frame_to_off = 0
        track_count = {}
        
        while True:
            # read current frame
            _, img = camera.read()

            frame_count += 1
            if frame_count % 5:
                # encode as a jpeg image and return it
                yield cv2.imencode('.jpg', img)[1].tobytes()
                continue

            if alerting:
                count_frame_to_off += 1
                if count_frame_to_off == ALERT // 2:
                    #Off
                    GPIO.output(PORT_PI, GPIO.LOW)
                    count_frame_to_off = 0
                    alerting = False
            
            _, buff = cv2.imencode('.jpg', img)
            jpg_as_text = base64.b64encode(buff)
            
            response = requests.post(api, json={'img': jpg_as_text}).json()

            recs = []
            face_info = json.loads(response['info'])
            # Draw bounding box
            for (class_id, conf, xmin, ymin, xmax, ymax) in face_info:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    rec = dlib.rectangle(xmin, ymin, xmax, ymax)
                    recs.append(rec)
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, 
                            "%s: %.2f" % (id2class[class_id], conf), 
                            (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, 
                            color)

            tracker_faces = FACE_TRACKERS.update(recs)
            print("z", len(tracker_faces), tracker_faces)

            for (faceID, centroid) in tracker_faces.items():
                print("id", faceID)
                if faceID not in track_count:
                    track_count[faceID] = 1
                else:
                    print("count", track_count[faceID])
                    track_count[faceID] += 1
                    if track_count[faceID] == ALERT:
                        #On
                        GPIO.output(PORT_PI, GPIO.HIGH)
                        alerting = True
                        print("heree")

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()