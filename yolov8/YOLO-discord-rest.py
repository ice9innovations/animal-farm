import mysql.connector
import hashlib
import socket

import os
import os.path

import requests
import json
import discord
from dotenv import load_dotenv
import uuid

from PIL import Image
from pathlib import Path

import base64
import time

load_dotenv()

API_URL = os.getenv('API_URL')
API_PORT = os.getenv('API_PORT')

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

HOST =  os.getenv('MYSQL_HOST')
USER =  os.getenv('MYSQL_USERNAME')
PW =  os.getenv('MYSQL_PASSWORD')
DB = os.getenv('MYSQL_DB')

mydb = mysql.connector.connect(
  host=HOST,
  user=USER,
  password=PW,
  database=DB,
  charset='utf8mb4'
)

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

async def post_to_API(dl_filename, message):
    qs = "/?path=./" + dl_filename
    url = API_URL + ":" + API_PORT + qs

    # do this before the API call or it will delete the image
    with open(dl_filename, 'rb') as imagefile:
        base64string = base64.b64encode(imagefile.read())
        img_hash = hashlib.new("sha3_256", base64string)
        digest = img_hash.hexdigest()

    #make API call
    response = requests.get(url)
    #print ("Querying API: " + url)
    json_results = json.loads(response.text)
    
    #print (response.text)
    #print (json_results)

    emos = []
    for data in json_results:
        emo = (data["emoji"])
        emos.append(emo)

    # only update db once for unique emojis, ignore duplicates
    emos = unique(emos)
    for emo in emos:
        print("Adding reaction: " + emo)
        update_database(message, str(digest), emo)

        await message.add_reaction(emo)
        time.sleep(.125)
    return

async def download_image(image_file, message, image_url):
    print(image_file)

    response = requests.get(image_file)

    dl_filename = uuid.uuid4().hex + ".jpg"
    open(dl_filename, "wb").write(response.content)

    await post_to_API(dl_filename, message)
    #await tnail(dl_filename, message, image_url)

    return dl_filename #image

def update_database(msg, image_url, reaction):
    try:
        bot = "YOLOv8"
        hostname = socket.gethostname()

        mycursor = mydb.cursor()

        sql = "INSERT INTO discord (server, channel, user, message, bot, image, reaction, hostname) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (msg.guild.id, msg.channel.id, msg.author.id, msg.id, bot, image_url, reaction, hostname)

        result = mycursor.execute(sql, val)
        mydb.commit()
    except:
        pass
    
intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)

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
    #print("Message received.")

    #if message.author.bot: # or message.channel.name != CHANNEL:
    #    return

    #print(`"CHANNELS")
    channel_access = False
    for CHANNEL in CHANNELS:
        if CHANNEL == message.channel.name:
            channel_access = True
            
    if channel_access:
        #if message.channel.name in CHANNEL:
        #if not message.content or len(message.attachments < 1):
            # You can use message.channel.send() if you don't want to mention user
            #await message.reply("Message should contain an image")
            #return
        #print ("Scanning attachments")

        for attachment in message.attachments:
            url = attachment.url
            response = "URL: " + url
            await download_image(url, message, url)

  #      if len(message.attachments) < 1:
            #print(len(message.attachments))
   #         url = message.attachments[0].url
    else:
        print("NO CHANNEL ACCESS")


async def reply(msg, body, discord_image):
    text = "Null"
    if (body):
        emu = await emoji(msg, body, discord_image)
        print(emu)
        text = body

    #await msg.channel.send(text)

#async def on_message(message):

client.run(TOKEN)
