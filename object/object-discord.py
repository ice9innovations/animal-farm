#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
# import the necessary packages
import numpy as np
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

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
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

async def classify(msg, img, image_url):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    image = cv2.imread(img)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
        (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

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
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            print(label)
                        
            await emoji(msg, label, image_url)
        
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output image
	#cv2.imshow("Output", image)
	#cv2.waitKey(0)

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
        await classify(message, tn_filename, image_url)

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





