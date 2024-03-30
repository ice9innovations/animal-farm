import asyncio

import ollama
from ollama import AsyncClient

import argparse

import discord
import json
import requests
import os
from dotenv import load_dotenv
import uuid
import base64

import sys

models = [
    "llama2",
    "llama2:13b",
    "llama2-uncensored",
    "codellama",
    "codellama:13b",
    "deepseek-coder:6.7b",
    "dolphin-mistral",
    "mistral",
    "mixtral",
    "phi",
    "wizard-vicuna-uncensored:13b",
    "llava"
]

global_model = "llama2"

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)

#ollama bridge

async def process_with_ollama(msg, img=""):
    prompt = msg.content
    #print(prompt)
    output = "Null"

    if (img):
        print ("Attachment received")
        prompt_lang = "In English, describe what's in this image: "
        prompt_image = os.path.join("/home/sd/ollama-api/",str(img))

        print (prompt_image)
        data = ""
        with open(img, 'rb') as image_file:
            data = base64.b64encode(image_file.read()).decode('utf-8')
            #print (data)

        #message = {'role': 'user', 'content': str(prompt)}
        print(prompt_lang)
        stream = await AsyncClient().generate(model='llava', prompt=prompt_lang,images=[data], stream=False)
        print (data)
        print (json.dumps(stream))
        #print (stream["response"])
        output = stream["response"]
        print(output)
        os.remove(img)
        await reply(output, msg)
        #await msg.channel.send(output)

    else:
        # Using Ollama to process and enhance the Pokémon details
        #print("Starting to process with ollama, note that this can take several minutes.")
        #llama2,codellama:13b,llama2-uncensored,mistral,dolphin-mistral,llama2:13b

        print(prompt)

        message = {'role': 'user', 'content': str(prompt)}
        use_model = "llama2"
        use_model = global_model
        response = await AsyncClient().chat(model=global_model,  messages=[message])

        output = response['message']['content']
        #await msg.channel.send(output)

        await reply(output, msg)

    #return response['message']['content']

async def download_image(message, image_url):
    print(image_url)
    response = requests.get(image_url)

    dl_filename = uuid.uuid4().hex + ".jpg"
    open(dl_filename, "wb").write(response.content)
    #await tnail(dl_filename, message, image_url)
    await process_with_ollama(message,dl_filename)

    return dl_filename #image

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
    if message.author.bot: # or message.channel.name != CHANNEL:
        author = str(message.author).lower()
        print (message.author)
        if not author.startswith("bernard"):
            return

    #prompt = message.content
    #print (prompt)

    channel_access = False
    for CHANNEL in CHANNELS:
        if CHANNEL == message.channel.name:
            channel_access = True
    if channel_access:
        attach = False
        for attachment in message.attachments:
            attach = True
            url = attachment.url
            response = "URL: " + url
            await download_image(message, url)

        if message.content.lower().startswith("/list"):
            await list_models(message)
        elif message.content.lower().startswith("/help"):
            await show_help(message)
        elif message.content.lower().startswith("/which"):
            await message.channel.send(global_model)
        elif message.content.lower().startswith("/use"):
            await select_model(message)
        else:
            if (not attach):
                await process_with_ollama(message, img="")
        #if message.channel.name == CHANNEL:
        #if not message.content or len(message.attachments < 1):
            # You can use message.channel.send() if you don't want to mention u>
            #await message.reply("Message should contain an image")
            #return

async def select_model(message):
    global global_model
    global models
    tmp_model = ""
    show_list = False

    set_model = message.content.lower().replace("/use ","")
    if (set_model.isnumeric()):
        # index is in range
        if 0 <= (int(set_model) - 1) < len(models):
            tmp_model = models[int(set_model) - 1]
            set_model = models[int(set_model) - 1]
    else:
        if set_model in models:
            tmp_model = set_model

    if tmp_model:
        global_model = set_model
        reply = "Setting model to: " + set_model
    else:
        reply = "Model not recognized."
        show_list = True

    await message.channel.send(reply)

    # show list of models if they got it wrong
    if show_list:
        await list_models(message)

async def list_models(message):
    global models
    reply = "Models:\n"

    cnt = 1
    for m in models:
        reply = reply + str(cnt) + ": " + m + "\n"
        cnt = cnt + 1

    #reply = "Models:\nllama2\nllama2:13b\nllama2-uncensored\ncodellama\ncodellama:13b\nmistral\ndolphin-mistral\nfalcon\nllava\nwizard-vicuna-uncensored:13b"
    await message.channel.send(reply)

async def show_help(message):
    reply = "Commands:\n"
    reply += "/list - List all avaialble models - Example \"\list\"\n"
    reply += "/use - Sets the model - Example \"use llama2\"\n"
    reply += "/which - Show the active model - Example \"which\""
    await message.channel.send(reply)

def split2len(s, n):
    def _f(s, n):
        while s:
            yield s[:n]
            s = s[n:]
    return list(_f(s, n))

async def reply(output, msg):
    print (output)
    output = output.replace("</SYS>","").replace("<SYS>","")

    if (len(output) > 0 and len(output) <= 2000):
        await msg.channel.send(output)
    else:
        msgs = split2len(output, 2000)
        for iter in msgs:
            await msg.channel.send(iter)

client.run(TOKEN)
