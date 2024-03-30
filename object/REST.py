from flask import Flask, request
#from gevent.pywsgi import WSGIServer

import os
import os.path

#import mysql.connector
import numpy as np
import cv2

import json
import requests
import os
from dotenv import load_dotenv
import uuid

from urllib.parse import urlparse, parse_qs

from PIL import Image

PRIVATE = False # change this for security to lmiit API to only local access
API_PORT = os.getenv('API_PORT')

#image_size = 384
image_size = 160

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

DEFAULT_CONFIDENCE = .2

# load our serialized model from disk
print("[INFO] loading model...")
prototext = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototext, model)

def tnail(img):
    try:
        tn_filename = uuid.uuid4().hex + ".jpg"
        image = Image.open(img)
        image.thumbnail((160,160))
        image.save(tn_filename)
        #image1 = Image.open('tn.png')
        #image1.show()
        return tn_filename
    except IOError:
        pass

def classify(img):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)

    tn = tnail(img)
    image = cv2.imread(tn)
    #image = cv2.imread(img)
    os.remove(tn) # delete thumbnail after use

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (160, 160)), 0.007843,
        (160, 160), 127.5)


    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # delete image file

    emoji_list = []
    key_list = []
    conf_list = []
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > DEFAULT_CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            print(label)

            emo = emoji(label)
            key_list.append(CLASSES[idx])
            conf_list.append(confidence * 100)
            emoji_list.append(emo)

    ret = ','.join(emoji_list)

    cnt = 0
    json_string = "["
    for key in key_list:
        
        #kk = key_json[key]
        #kv = key_json[value]
        tmp = '{"object": "' + key + '", "confidence": ' + str(round(conf_list[cnt],3)) + ', "emoji": "' + emoji_list[cnt] + '"},'
        json_string = json_string + tmp
        cnt = cnt + 1

    json_string = json_string.rstrip(',')
    json_string = json_string + "]"

    print(emoji_list)
    return json_string
        
	# show the output image
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)
 
def emoji(preds):
    print(preds)
    emojis = []
    #print(preds)
    #tags = preds.split(",")

    max = 3
    count = 0
    threshold = .33

    print(preds)
    #current_emoji = 
    aPred = preds.split(":")
    
    tag = aPred[0]
    tag = tag.strip()

    tmp = aPred[1]
    tmp = tmp.strip().replace("%","")
    strength = float(tmp)

    print("Strength: " + str(strength))
    
    #if (strength > threshold):
    emo = lookup_emoji(tag)
    return emo
        #print(current_emoji)

def lookup_emoji(tag):
    f = open('./emojis.json')
    data = json.load(f)

    print("Tag: " + tag )
    for d in data:
        for key, value in d.items():
            #print(key, value)
            if (value == tag or key == tag):
                print(key, value)
                if (key):
                    #update_database(msg, image_url, value)
                    #msg.add_reaction(value)
                    return value
                #return value


# create folder for uploaded data
FOLDER = './'
#os.makedirs(FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        url = request.args.get('url')
        path = request.args.get('path')

        if url:
            print (url)
            fn =  uuid.uuid4().hex + ".jpg"

            response = requests.get(url)
            with open(fn, mode="wb") as file:
                file.write(response.content)

            emojis = classify(fn)
            os.remove(fn)

            return emojis
        elif path:
            emojis = classify(path)
            os.remove(path)

            return emojis
        else:
            #default, basic HTML
            
            html = ""
            try:
                with open('form.html', 'r') as file:
                    html = file.read()
            except:
                html = '''<form enctype="multipart/form-data" action="" method="POST">
                    <input type="hidden" name="MAX_FILE_SIZE" value="8000000" />
                    <input name="uploadedfile" type="file" /><br />
                    <input type="submit" value="Upload File" />
                </form>'''
                    
            return html
        
    if request.method == 'POST':
        emojis = []
        for field, data in request.files.items():
            fn = uuid.uuid4().hex + ".jpg"
            print('field:', field)
            print('filename:', data.filename)
            print('UUID:', fn)
            if data.filename:
                data.save(os.path.join(FOLDER, fn)) #data.filename
                emojis = classify(fn)
                os.remove(fn)

        #return "OK"
        return emojis

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    if PRIVATE:
        IP = "127.0.0.1"

    app.run(use_reloader=False, debug=True, host=IP, port=7777)

    # Production
    # Requires a different server to be installed
    #http_server = WSGIServer((IP, API_PORT), app)
    #http_server.serve_forever()