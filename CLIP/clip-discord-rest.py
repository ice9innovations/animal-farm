import mysql.connector
import hashlib
import base64
import socket

import discord
import json
import requests
import os
import uuid

from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv('API_URL')
API_PORT = os.getenv('API_PORT')

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
if CHANNELS:
    CHANNELS = CHANNELS.split(",")
else:
    CHANNELS = []

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

async def post_to_API(dl_filename, message):
    qs = "/?path=./" + dl_filename
    url = API_URL + ":" + API_PORT + qs

    # This must be done before the API call is made
    # or the image will be deleted
    with open(dl_filename, 'rb') as imagefile:
        base64string = base64.b64encode(imagefile.read())
        img_hash = hashlib.new("sha3_256", base64string)
        digest = img_hash.hexdigest()
    
    response = requests.get(url)
    #print ("Querying API: " + url)
    #print ("Faces:")
    json_results = json.loads(response.text)
    #print(js)

    #print (json_results)

    for data in json_results["CLIP"]:
        #print (data)
        #try:
        for item in data:
            #emo = item["emoji"]
            if item == "emoji":
                emoji = data[item]
                #print (emojis)
                #emoji_list = emojis.split(",")
                #for emo in emoji_list:
                print ("Adding reaction: " + emoji)
                update_database(message, digest, emoji)

                await message.add_reaction(emoji)

    #print(response.text)
    return

async def download_image(url, message):
    #print(image_file)
    dl_filename = uuid.uuid4().hex + ".jpg"

    response = requests.get(url)
    with open(dl_filename, mode="wb") as file:
        file.write(response.content)

    await post_to_API(dl_filename, message)
    return dl_filename #image

def update_database(msg, image_url, reaction):
    try:
        bot = "CLIP"
        hostname = socket.gethostname()

        mycursor = mydb.cursor()

        sql = "INSERT INTO discord (server, channel, message, user, bot, image, reaction, hostname) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (msg.guild.id, msg.channel.id, msg.id, msg.author.id, bot, image_url, reaction, hostname)

        mycursor.execute(sql, val)

        mydb.commit()
    except:
        pass

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
            await download_image(url, message)

  #      if len(message.attachments) < 1:
            #print(len(message.attachments))
   #         url = message.attachments[0].url

client.run(TOKEN)





