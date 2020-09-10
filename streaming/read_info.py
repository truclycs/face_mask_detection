import __init__
import os
import json


f = open(os.path.join('streaming', 'info.json'), "r")
data = json.load(f)
IP = data['IP']
port = data['PORT']
camera_source = data['Video_Source']
if camera_source == '0':
    camera_source = 0
api = data['API']