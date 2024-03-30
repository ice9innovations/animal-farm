#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
# import the necessary packages
from flask import Flask, request

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

import discord
import json
import requests
import os
from dotenv import load_dotenv
import uuid

from urllib.parse import urlparse, parse_qs
from PIL import Image

PRIVATE = False # change this for security to limit API requests to only local access
API_PORT = os.getenv('API_PORT')

#image_size = 384
image_size = 160

#mydb = mysql.connector.connect(
#  host=HOST,
#  user=USER,
#  password=PW,
#  database=DB,
#  charset='utf8mb4'
#)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def classify(img):

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img) #args["image"])
    image = imutils.resize(image, width=image_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    try:
        os.remove(img)
    except:
        print("Could not remove image: " + img)

    cnt = 0
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        cnt = cnt + 1
    
    ret = ""
    ret = '{ "faces": ' + str(cnt) + '}'
    json_ret = json.dumps(ret)

    return ret

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)
   
def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()


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

    app.run(debug=True, host="0.0.0.0", port=7772)


