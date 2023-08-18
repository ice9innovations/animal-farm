#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
import pytesseract
import argparse
import cv2
import numpy as np
import re
import math

import discord
import json
import requests
import os
from dotenv import load_dotenv
import uuid

from PIL import Image

from exif import Image as ExifImage

#image_size = 384
image_size = 240

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

intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)


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


async def tnail(img, message, image_url):
    try:
        tn_filename = uuid.uuid4().hex + ".jpg"

        image = Image.open(img)
        image.thumbnail((240,240))
        image = image.convert('RGB')

        exif_data = ExifImage(image)

        exif_data.resolution = 70
        exif_data.XResolution = 70
        exif_data.YResolution = 70
        print(exif_data)


        #image.save(tn_filename)

        os.remove(img) # delete the original image

        await classify(exif_data, message, image_url)

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

    await classify(dl_filename, message, image_url)
    #await tnail(dl_filename, message, image_url)

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

async def classify(image_path, message, image_url):
    
    # load the input image and convert it from BGR to RGB channel
    # ordering}
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image)

    #osd = pytesseract.image_to_osd(image)
    #angle = re.search('(?<=Rotate: )\d+', osd) #.group(0)
    #script = re.search('(?<=Script: )\d+', osd) #.group(0)

    #print("angle: ", angle[0])
    #print("script: ", script)

    #deg = angle[0]
    #deg = int(deg) * -1

    #image = rotation(image, deg)

    #gray  = None
    #thresh = None
    #opening = None
    #canny = None

    #if (image is not None):
    #    gray = get_grayscale(image)
    #    thresh = thresholding(gray)
    #    opening = opening(gray)
    #    canny = canny(gray)

    # use Tesseract to OCR the image
    text = None
    text = pytesseract.image_to_string(image)
    print(text)

    print(image_path)
    os.remove(image_path) # delete the original image

    if (text != None):
        await message.add_reaction("💬")

        #await reply(message, "text", image_url)
        #await reply(message, ret, image_url)


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



