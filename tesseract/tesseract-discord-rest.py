#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

import mysql.connector
import hashlib
import socket
import base64

import re

import discord
import json
import requests
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

#image_size = 384
image_size = 240

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

intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)

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

    response = requests.get(url)
    #print(response.text)

    text = str(response.text).replace("\n"," ").strip()
    json_results = json.loads(text)
    print (json_results)

    emos = []
    texts = []
    for data in json_results:
        emo = (data["emoji"])
        txt = data["text"]
        emos.append(emo)
        texts.append(txt)
        await message.add_reaction(emo)

    # only update db once for unique emojis, ignore duplicates
    cnt = 0
    emos = unique(emos)
    for emo in emos:
        txt = texts[cnt]
        print("Update database: " + emo + " " + txt)
        update_database(message, str(digest), emo, txt)
        cnt = cnt + 1

    return

def update_database(msg, image_url, reaction, text):
    try:
        bot = "Tesseract"
        mycursor = mydb.cursor()

        hostname = socket.gethostname()
        sql = "INSERT INTO discord (server, channel, user, message, bot, image, text, reaction, hostname) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (msg.guild.id, msg.channel.id, msg.author.id, msg.id, bot, image_url, text, reaction, hostname)
        
        result = mycursor.execute(sql, val)
        mydb.commit()
    except:
        pass

    #print(result)

async def download_image(image_file, message):
    response = requests.get(image_file)

    dl_filename = uuid.uuid4().hex + ".jpg"
    open(dl_filename, "wb").write(response.content)

    await post_to_API(dl_filename, message)
    return dl_filename #image

async def stripspaces(input):
    output = str(re.sub(r"[\n\t\s]*", "", input)).strip()
    return output

async def tagImageFromAPI(url):
    url = "http://127.0.0.1:7777?url=" + url
    print (url)

    response = requests.get(url)
    print (response)

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
            #await download_image(url, message, url)
            await download_image(url, message)

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



