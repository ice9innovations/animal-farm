#https://discord.com/oauth2/authorize?client_id=1122009842333257758&permissions=274878285888&scope=bot%20applications.commands%20messages.read

#import mysql.connector
import uuid

import discord
import os
from dotenv import load_dotenv

import requests
import json

import torch

from urllib.parse import urlparse, parse_qs

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

import sys

import socketserver # Establish the TCP Socket connections

image_size = 384
#image_size = 160

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

base_dir = "/home/niro/BLIP/"
model_url = base_dir + 'model_base_14M.pth'
model_url = base_dir + 'model_large.pth'
model_url = base_dir + 'model_base_capfilt_large.pth'
model_url = base_dir + 'model_base.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

async def emoji(msg, preds, image_url):
    emojis = []
    #print(preds)
    tags = preds.split(" ")

    #count = 0
    for tag in tags:
        if (tag):
            print(tag)
                
            await lookup_emoji(msg, tag, image_url)
            #if (count <= max):
            #    count = count + 1

                
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

def download_image(image_file,image_size,device):
    raw_image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    #raw_image = Image.open(open(img_file, "rb")).convert('RGB')

    w,h = raw_image.size
    #print(image_file)
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    #image = image.view(2, 2)
    return image

async def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "BLIP", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

async def classify(img, msg):
        
        #print(img)

        #sys.argv[1]
        #image = download_image(img,msg)
        image = download_image(img,image_size=image_size, device=device)

        with torch.no_grad():
            # beam search
            caption = model.generate(image, sample=False, num_beams=4, max_length=20, min_length=5)
            # nucleus sampling
            #caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

            print('caption: '+caption[0])
            await reply(msg, caption[0], img)


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
            await classify(url, message)


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
