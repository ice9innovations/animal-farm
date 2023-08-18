#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
# import the necessary packages
from imutils import face_utils
import numpy as np
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

#image_size = 384
image_size = 160

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

HOST =  os.getenv('MYSQL_HOST')
USER =  os.getenv('MYSQL_USERNAME')
PW =  os.getenv('MYSQL_PASSWORD')
DB = os.getenv('MYSQL_DB')

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

async def classify(msg, img):

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img) #args["image"])
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        os.remove(img)
        # show the face number
        print("Face #{}".format(i + 1), (x - 10, y - 10))
        #await update_database(msg, image_url, value)
        await msg.add_reaction("🧑")


        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            # show the output image with the face detections + facial landmarks
    #cv2.imshow("Output", image)
    #cv2.waitKey(0)

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


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

async def tnail(img, message, image_url):
    try:
        tn_filename = uuid.uuid4().hex + ".jpg"

        image = Image.open(img)
        image.thumbnail((160,160))
        image.save(tn_filename)
        #image1 = Image.open('tn.png')
        #image1.show()

        os.remove(img) # delete the original image

        #await classify(tn_filename, message, image_url)
        await classify(message, tn_filename)

    except IOError:
        pass
   
async def download_image(image_file, message, image_url):
    print(image_file)

    #raw_image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    #raw_image = Image.open(open(img_file, "rb")).convert('RGB')
    #w,h = raw_image.size
    #transform = transforms.Compose([
    #    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #    ])
    #image = transform(raw_image).unsqueeze(0).to(device)
    
    response = requests.get(image_file)

    dl_filename = uuid.uuid4().hex + ".jpg"
    open(dl_filename, "wb").write(response.content)
    await tnail(dl_filename, message, image_url)

    return dl_filename #image


async def emoji(msg, preds, image_url):
    emojis = []
    #print(preds)
    #tags = preds.split(",")

    max = 3
    count = 0
    threshold = .33

    for pred in preds:
        if (pred):
            if (count <= max):
                count = count + 1

                print(pred)
                #current_emoji = 
                aPred = pred.split(":")
                
                tag = aPred[0]
                strength = float(aPred[1])

                print("Strength: " + str(strength))
                
                if (strength > threshold):
                    await lookup_emoji(msg, tag, image_url)
                    #print(current_emoji)

async def lookup_emoji(msg, tag, image_url):
    f = open('./emojis.json')
    data = json.load(f)

    print("Tag: " + tag )
    for d in data:
        for key, value in d.items():
            #print(key, value)
            if (value == tag or key == tag):
                print(key, value)
                if (key):
                    await update_database(msg, image_url, value)
                    await msg.add_reaction(value)

                #return value

async def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

async def reply(msg, body):
    text = "Null"
    if (body):
        emu = await emoji(msg, body)
        print(emu)
        text = body


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

@client.event
async def on_message(message):
    #if message.author.bot: # or message.channel.name != CHANNEL:
    #    return

    channel_access = False
    for CHANNEL in CHANNELS:
        if CHANNEL == message.channel.name:
            channel_access = True
            
    if channel_access:

        #if message.channel.name == CHANNEL:
        #if not message.content or len(message.attachments < 1):
            # You can use message.channel.send() if you don't want to mention user
            #await message.reply("Message should contain an image")
            #return

        for attachment in message.attachments:
            url = attachment.url
            response = "URL: " + url
            await download_image(url, message, url)

  #      if len(message.attachments) < 1:
            #print(len(message.attachments))
   #         url = message.attachments[0].url


async def reply(msg, body, image_url):
    text = "Null"
    if (body):
        emu = await emoji(msg, body, image_url)
        print(emu)
        text = body

    #await msg.channel.send(text)

#async def on_message(message):

client.run(TOKEN)





