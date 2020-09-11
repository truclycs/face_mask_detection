import __init__
import os
from flask import Flask, render_template, request, redirect
import base64
import urllib.request
import io
import numpy as np
from PIL import Image
import json

from ailibs.dectector.ssd.FaceMaskDetector import FaceMaskDetector


face_mask_detector = FaceMaskDetector(
    model_path=os.path.join('ailibs_data', 'model360.pth'),
    log=True
)


app = Flask(__name__)


def get_prediction(image_base64):
    """Get infomation of all face mask in the image from detector

    Args:
        image_base64: image in base64

    Returns:
        result (dict): information of faces (id, conf, xmin, ymin, xmax, ymax)

    """    
    imgdata = base64.b64decode(image_base64)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    image = Image.open(io.BytesIO(imgdata))
    return face_mask_detector.detect(np.array(image))


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img_str = request.json['img']
        return get_prediction(img_str)


if __name__ == '__main__':
    f = open(os.path.join('apis', 'host_info.json'), "r")
    data = json.load(f)
    IP = data['IP']
    port = data['PORT']
    app.run(host=IP, port=port, threaded=True)