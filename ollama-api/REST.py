# Copyright (c) Facebook, Inc. and its affiliates.
import multiprocessing as mp
import os
import asyncio
import base64
import string

#import ollama
from ollama import AsyncClient

import json
import requests
import os
import uuid

from flask import Flask, request
from dotenv import load_dotenv

from urllib.parse import urlparse, parse_qs

from PIL import Image

# constants
WINDOW_NAME = "COCO detections"

load_dotenv()
FOLDER = './'

IMAGE_SIZE = 160

HOST =  os.getenv('MYSQL_HOST')
USER =  os.getenv('MYSQL_USERNAME')
PW =  os.getenv('MYSQL_PASSWORD')
DB = os.getenv('MYSQL_DB')


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

async def process_with_ollama(img="", param_model=global_model):
    #print ("Process with oLLaMa")
    #prompt = msg.content
    #print(prompt)
    #print(img)
    output = "Null"

    if (img):
        #print("Process Image: " + img)
        #print ("Attachment received")
        prompt_lang = "Write a single short sentence to describe this image as simply as possible." #"as a caption for an image: "
        prompt_image = os.path.join("/home/sd/ollama-api/",str(img))
        #prompt_system = "You are an image captioner for innumerate people. Use only short, non-descriptive language and do NOT use numbers." # Don't use numbers. Use only words. Do not count. Do not estimate age. Do not estimate dates. Do not estimate."
        prompt_system = ""

        print(prompt_image)
        #print (prompt_image)
        data = ""
        with open(img, 'rb') as image_file:
            data = base64.b64encode(image_file.read()).decode('utf-8')
            #print (data)

        #message = {'role': 'user', 'content': str(prompt)}
        #print(prompt_lang)
        stream = await AsyncClient().generate(model='llava', system=prompt_system, prompt=prompt_lang,images=[data], stream=False)
        #stream = await AsyncClient().generate(model='bakllava', system=prompt_system, prompt=prompt_lang,images=[data], stream=False)
        #print (data)
        #print (json.dumps(stream))
        #print (stream["response"])
        output = stream["response"]
        #print(output)
        os.remove(img)
        
        #await reply(output, msg)
        #await msg.channel.send(output)

    else:
        # Using Ollama to process and enhance the Pokémon details
        #print("Starting to process with ollama, note that this can take several minutes.")
        #llama2,codellama:13b,llama2-uncensored,mistral,dolphin-mistral,llama2:13b

        #print(prompt)

        message = {'role': 'user', 'content': str(prompt)}
        use_model = "llama2"
        use_model = global_model
        response = await AsyncClient().chat(model=param_model,  messages=[message])

        output = response['message']['content']
        #await msg.channel.send(output)

    return output

    #return response['message']['content']


def emoji(preds):
    emojis = []
    #print(preds)
    #tags = preds.split(",")

    max = 3
    count = 0
    threshold = .33

    emojis = []
    for pred in preds:
        if (pred):
            if (count <= max):
                count = count + 1

                #print(pred)
                #current_emoji = 
                aPred = pred.split(":")
                
                tag = aPred[0]
                strength = aPred[1]

                print("Tag: " + str(tag))
                print("Strength: " + str(strength))
                emo = lookup_emoji(tag)
                emojis.append(emo)

                #if (strength > threshold):
                    #await lookup_emoji(msg, tag, image_url)
                    #print(current_emoji)
                
    return emojis


def lookup_emoji(tag):
    f = open('./emojis.json')
    data = json.load(f)

    #print("Tag: " + tag )
    for d in data:
        for key, value in d.items():
            #print(key, value)
            if (value == tag or key == tag):
                #print(key, value)
                #if (key):
                    #update_database(value)
                    #await msg.add_reaction(value)

                return value

    
async def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        url = request.args.get('url')
        path = request.args.get('path')

        if url:
            #print (url)
            fn = uuid.uuid4().hex + ".jpg"

            response = requests.get(url)
            with open(fn, mode="wb") as file:
                file.write(response.content)

            detect = asyncio.run(process_with_ollama(fn))
            detect = detect.translate(str.maketrans('', '', string.punctuation))

            ret = '{ "caption": "' + detect.replace("\n"," ").strip() + '"}'

            print(ret)

            return ret
        elif path:
            detect = asyncio.run(process_with_ollama(fn))
            detect = detect.translate(str.maketrans('', '', string.punctuation))

            ret = '{ "caption": "' + detect + '"}'
            print(ret)

            return ret
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
        for field, data in request.files.items():
            fn = uuid.uuid4().hex + ".jpg"
            print('field:', field)
            print('filename:', data.filename)
            print('UUID:', fn)
            data.save(os.path.join(FOLDER, fn)) #data.filename

            if data.filename:
                detect = asyncio.run(process_with_ollama(fn))
                detect = detect.translate(str.maketrans('', '', string.punctuation))

                ret = '{ "caption": "' + detect + '"}'

                print(ret)

        #return "OK"
        return ret

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    IP = "127.0.0.1"
    app.run(use_reloader=False,debug=True, host="0.0.0.0", port=7781)
