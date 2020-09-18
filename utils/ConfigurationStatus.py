import os
import json
import django
# from __init__ import PYTHON_PATH
os.environ['DJANGO_SETTINGS_MODULE'] = 'checkin.settings'

from utils.ImageStorage import ImageStorage
from checkin.facecheckin.models import Employee, FaceImage, CheckInTime, Configuration, PretrainedImage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from django.conf import settings
STORAGE_LOCATION = getattr(settings, 'STORAGE_LOCATION')
IMAGESTORAGE = ImageStorage(STORAGE_LOCATION)

STREAMING_NOT_DEFINED = "none"
CHECKIN_DURATION = 30 #seconds

class TRAINING_STATUS:
    START = "start"
    IN_PROCESSING = "in_processing"
    STOP = "stop"


class UPDATED_FACE_CLASSIFIER:
    TRUE = "true"
    FALSE = "false"


class CAMERA_CONFIG:
    CAMERA_URL = 0
    USER_ID = STREAMING_NOT_DEFINED
        
    def update():
        CAMERA_CONFIG.CAMERA_URL = Configuration.objects.get(key="camera_url").value
        if CAMERA_CONFIG.CAMERA_URL == "0":
            CAMERA_CONFIG.CAMERA_URL = 0
        CAMERA_CONFIG.USER_ID = Configuration.objects.get(key="streaming_user_id").value

    def deregister():
        Configuration.objects.filter(key="streaming_user_id").update(value=STREAMING_NOT_DEFINED)


def update_db(info, face_imag):
    try:
        print("FaceImage", info['employee_id'])
        image = FaceImage()
        image.save()
        print("FaceImage", info['employee_id'], getattr(image, "id"))
        IMAGESTORAGE.write(face_imag, getattr(image, "id"))

        info['image_id'] = getattr(image, "id")
        records = CheckInTime.objects.filter(employee__id=info['employee_id'], end_time__gte=info['end_time']-CHECKIN_DURATION).update(end_time=info['end_time'])
        if not records:
            print("Creating...", info)
            # checkin = CheckInTime(**info)
            emp =  Employee.objects.get(id=info['employee_id'])
            if not emp:
                raise Exception("Invalid Employee {}".format(info['employee_id']))

            checkin = CheckInTime(employee_id=info['employee_id'],
                                image_id=info['image_id'],
                                start_time=info['start_time'],
                                end_time=info['end_time'])
            checkin.save()
            print(">>>Creating CheckInTime", info['employee_id'], getattr(checkin, "start_time"))
        else: 
            print(">>>Updating CheckInTime")
    except Exception as e:
        print(e)