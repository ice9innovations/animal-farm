import json
import requests
import os
import sys
import uuid

from dotenv import load_dotenv

import torch

from urllib.parse import urlparse, parse_qs

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

from flask import Flask, request

image_size = 384
#image_size = 160

load_dotenv()
FOLDER = './'

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

HOST =  os.getenv('MYSQL_HOST')
USER =  os.getenv('MYSQL_USERNAME')
PW =  os.getenv('MYSQL_PASSWORD')
DB = os.getenv('MYSQL_DB')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")

#model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

base_dir = FOLDER
model_url = base_dir + 'model_base_14M.pth'
model_url = base_dir + 'model_large.pth'
model_url = base_dir + 'model_base_capfilt_large.pth'
model_url = base_dir + 'model_base.pth'

model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

def emoji(preds):
    emojis = []
    #print(preds)
    tags = preds.split(" ")

    #count = 0
    for tag in tags:
        if (tag):
            #print(tag)
                
            emo = lookup_emoji(tag)
            if (emo):
                emojis.append(emo)
            #if (count <= max):
            #    count = count + 1

                
                #print(current_emoji)
    return emojis

def lookup_emoji(tag):
    f = open('./emojis.json')
    data = json.load(f)

    #print("Tag: " + tag )
    emoji_list = []
    for d in data:
        for key, value in d.items():
            #print(key, value)
            if (value == tag or key == tag):
                #print(key, value)
                if (key):
                    #update_database(msg, image_url, value)
                    return value

def preprocess(image_file):   
    global device
 
    print (image_file)
    #raw_image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    #im = open(image_file, "rb")
    raw_image = Image.open(image_file).convert('RGB')

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

def buildJSON(caption, emoji_list):
    emoji_str = ",".join(emoji_list)

    ret = '{\n  "BLIP": [\n'
    ret = ret + '    { "caption": "' + caption + '" },\n'
    ret = ret + '    { "emojis": "' + emoji_str + '" }\n'
    ret = ret + '  ]\n}'

    return ret

def classify(img):
    global device

    #print(img)

    #sys.argv[1]
    image = preprocess(img)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=1, max_length=20, min_length=5)
        # nucleus sampling
        #caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)

        print('caption: '+caption[0])
        #await reply(msg, caption[0], img)

    emoji_list = emoji(caption[0])
    #print(emoji_list)
    ret = buildJSON(caption[0],emoji_list)

    os.remove(img)
    return ret

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        url = request.args.get('url')
        path = request.args.get('path')

        if url:
            print (url)
            fn = uuid.uuid4().hex + ".jpg"

            response = requests.get(url)
            with open(fn, mode="wb") as file:
                file.write(response.content)

            BLIP = classify(fn)

            print(BLIP)

            return BLIP
        elif path:
            BLIP = classify(path)
            print(BLIP)

            return BLIP
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
                BLIP = classify(fn)
                print(BLIP)

        #return "OK"
        return BLIP

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    IP = "127.0.0.1"
    app.run(use_reloader=False,debug=True, host="0.0.0.0", port=7777)
