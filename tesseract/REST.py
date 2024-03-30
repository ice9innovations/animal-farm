from flask import Flask, request
#from gevent.pywsgi import WSGIServer

import os
import os.path

#import mysql.connector

import pytesseract
import cv2
import numpy as np
import re
import math

import json
import requests
import os
import os.path

import re
import uuid

from dotenv import load_dotenv
import uuid

PRIVATE = os.getenv('PRIVATE') # change this for security to limit API requests to only local access
API_PORT = os.getenv('API_PORT')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def rotation(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


def stripspaces(input):
    output = str(re.sub(r"[\n\t\s]*", "", input)).strip()
    return output

def classify(image_path):
    
    # load the input image and convert it from BGR to RGB channel
    # ordering}
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert to grayscale
    image = get_grayscale(image)
    image = remove_noise(image)
    image = deskew(image)
    #image = await canny(image)

    #print(image)

    #osd = pytesseract.image_to_osd(image)
    #angle = re.search('(?<=Rotate: )\d+', osd) #.group(0)
    #script = re.search('(?<=Script: )\d+', osd) #.group(0)

    #print("angle: ", angle[0])
    #print("script: ", script)

    #deg = angle[0]
    #deg = int(deg) * -1

    #image = rotation(image, deg)

    #if (image is not None):
    #    gray = get_grayscale(image)
    #    thresh = thresholding(gray)
    #    opening = opening(gray)
    #    canny = canny(gray)

    # use Tesseract to OCR the image
    text = None

    ret = "" # return value
    try:
        text = pytesseract.image_to_string(image)
        #text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

        #print(image_path)

        stripped_text = stripspaces(text)
        if (stripped_text):
            print("Text: " + text)
            txt = ''.join(text)
            txt = txt.replace("\n"," ")
            txt = re.sub('\W+',' ',txt).strip()

            #txt = txt.strip()

            ret = '[{"text": "' + txt + '", "emoji": "💬"}]'
            #ret = json.dumps(ret)
            #await message.add_reaction("💬")

            #await reply(message, "text", image_url)
            #await reply(message, ret, image_url)
        else:
            ret = '[{"text": "", "emoji": ""}]'
            print("Text: none")

    except:
        print ("Exception")

    os.remove(image_path) # delete the original image
    return ret

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

            return emojis
        elif path:
            emojis = classify(path)

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

        #return "OK"
        return emojis

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    if PRIVATE:
        IP = "127.0.0.1"

    app.run(debug=True, host=IP, port=7775)

    # Production
    # Requires a different server to be installed
    #http_server = WSGIServer((IP, API_PORT), app)
    #http_server.serve_forever()
