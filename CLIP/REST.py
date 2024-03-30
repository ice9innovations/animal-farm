from flask import Flask, request
#from gevent.pywsgi import WSGIServer

#import tensorflow as tf

import torch
import torchvision.transforms as transforms
from PIL import Image
import clip
import sys
import numpy as np

import os
import os.path

import requests
import json
import discord

import re
import uuid

from dotenv import load_dotenv


load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

# device CUDA, Silicon, or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = torch.device("mps")

## SETUP data environment

#coco
coco = "diagram","chart","graph","flower","person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"

#flowers
flowers = "rose","sunflower","hydrangea","lavender","peony","daisy","orchid","tulip","lily","chrysanthemum","dahlia","daffodil","iris","periwinkle","azalea","calla lily","carnation","marigold","violet","begonia","gardenia","hyacinth","kale","amaranthus","geranium","gladiolus","petunia","ranunculus","zinnia","anemone","aster","black-eyed susan","buttercup","larkspur","pansy","snapdragon","delphinium","gerbera","daisy","yarrow","allium","astilbe","cornflower","crocus","dianthus","phlox","alstroemeria","poppy","celosia","agapanthus","lisianthus","freesia","campanula","dusty miller","liatris","gypsophila","scabiosa","hypericum","eryngium","brunia"

#objectnet
objectnet = "text","man","woman","child","car","truck","boat","Air freshener","Alarm clock","Backpack","Baking sheet","Banana","Bandaid","Baseball bat","Baseball glove","Basket","Bathrobe","Bath towel","Battery","Bed sheet","Beer bottle","Beer can","Belt","Bench","Bicycle","Bike pump","Bills (money)","Binder (closed)","Biscuits","Blanket","Blender","Blouse","Board game","Book (closed)","Bookend","Boots","Bottle cap","Bottle opener","Bottle stopper","Box","Bracelet","Bread knife","Bread loaf","Briefcase","Brooch","Broom","Bucket","Butcher’s knife","Butter","Button","CD/DVD case","Calendar","Can opener","Candle","Canned food","Cellphone","Cellphone case","Cellphone charger","Cereal","Chair","Cheese","Chess piece","Chocolate","Chopstick","Clothes hamper","Clothes hanger","Coaster","Coffee beans","Coffee grinder","Coffee machine","Coffee table","Coin (money)","Comb","Combination lock","Computer mouse","Contact lens case","Cooking oil bottle","Cork","Cutting board","DVD player","Deodorant","Desk lamp","Detergent","Dishrag or hand towel","Dish soap","Document folder (closed)","Dog bed","Doormat","Drawer (open)","Dress","Dress pants","Dress shirt","Dress shoe (men)","Dress shoe (women)","Drill","Drinking Cup","Drinking straw","Drying rack for clothes","Drying rack for plates","Dust pan","Earbuds","Earring","Egg","Egg carton","Envelope","Eraser (white board)","Extension cable","Eyeglasses","Fan","Figurine or statue","First aid kit","Flashlight","Floss container","Flour container","Fork","French press","Frying pan","Glue container","Hair brush","Hair clip","Hairdryer","Hair tie","Hammer","Hand mirror","Handbag","Hat","Headphones (over ear)","Helmet","Honey container","Ice","Ice cube tray","Iron","Ironing board","Jam","Jar","Jeans","Kettle","Keyboard","Key chain","Ladle","Lampshade","Laptop (open)","Laptop charger","Leaf","Leggings","Lemon","Letter opener","Lettuce","Light bulb","Lighter","Lipstick","Loofah","Magazine","Makeup","Makeup brush","Marker","Match","Measuring cup","Microwave","Milk","Mixing/Salad Bowl","Monitor","Mousepad","Mouthwash","Mug","Multitool","Nail","Nail clippers","Nail file","Nail polish","Napkin","Necklace","Newspaper","Nightlight","Nightstand","Notebook","Notepad","Nut for a screw","Orange","Oven mitts","Padlock","Paintbrush","Paint can","Paper","Paper bag","Paper plates","Paper towel","Paperclip","Peeler","Pen","Pencil","Pepper shaker","Pet food container","Landline phone","Photograph","Pill bottle","Pill organizer","Pillow","Pitcher","Placemat","Plastic bag","Plastic cup","Plasticwrap","Plate","Playing cards","Pliers","Plunger","Pop can","Portable heater","Poster","Power bar","Power cable","Printer","Raincoat","Rake","Razor","Receipt","Remote control","Removable blade","Ribbon","Ring","Rock","Rolling pin","Ruler","Running shoe","Safety pin","Salt shaker","Sandal","Scarf","Scissors","Screw","Scrub brush","Shampoo bottle","Shoelace","Shorts","Shovel","Skateboard","Skirt","Sleeping bag","Slipper","Soap bar","Soap dispenser","Sock","Soup Bowl","Sewing kit","Spatula","Speaker","Sponge","Spoon","Spray bottle","Squeegee","Squeeze bottle","Standing lamp","Stapler","Step stool","Still Camera","Sink Stopper","Strainer","Stuffed animal","Sugar container","Suit jacket","Suitcase","Sunglasses","Sweater","Swimming trunks","T-shirt","TV","Table knife","Tablecloth","Tablet","Tanktop","Tape","Tape measure","Tarp","Teabag","Teapot","Tennis racket","Thermometer","Thermos","Throw pillow","Tie","Tissue","Toaster","Toilet paper roll","Tomato","Tongs","Toothbrush","Toothpaste","Tote bag","Toy","Trash bag","Trash bin","Travel case","Tray","Trophy","Tweezers","Umbrella","USB cable","USB flash drive","Vacuum cleaner","Vase","Video camera","Walker","Walking cane","Wallet","Watch","Water bottle","Water filter","Webcam","Weight (exercise)","Weight scale","Wheel","Whisk","Whistle","Wine bottle","Wine glass","Winter glove","Wok","Wrench","Ziploc bag"

#convert to lowercase
objectnet_lower = []
for string in objectnet:
    objectnet_lower.append(string.lower())
objectnet_lower = tuple(objectnet_lower)

#merge labels
labels = coco + objectnet_lower + flowers

labels_desc = []
for label in labels:
    label = "a picture of a " + label
    labels_desc.append(label)

label_tensor = clip.tokenize(labels_desc).to(device)

model, preprocess = clip.load("ViT-B/32", device=device)


def lookup_emoji(tag, score):
    f = open('./emojis.json')
    data = json.load(f)

    print("TAG: " + tag)
    emoji_list = []
    for d in data:
        for key, value in d.items():
            #print(key, value)
            tag = tag.strip()

            clean_tag = tag.replace(",","").strip() # drop extra comma

            if (value == clean_tag or key == clean_tag):
                #print("TAG: " + clean_tag)
                print(key, value)

                if value:
                    #await update_database(msg, discord_image, value)
                    emoji_list.append(key + " | " + value + " | " + str(round(score,3)))
                    #await msg.add_reaction(value)

    emoji_list = unique(emoji_list)
    ret = ','.join(emoji_list)
    return ret
                #return value
                        

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

def compute_similarity(image_path, model, label_tensor):
    with torch.no_grad():
        image_tensor = preprocess_image(image_path)
        image_features = model.encode_image(image_tensor)
        label_features = model.encode_text(label_tensor)
        similarity = (image_features @ label_features.T).softmax(dim=-1)
    return similarity


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def buildJSON(emoji_list):
    spl = emoji_list.split(" | ")
    key = spl[0]
    emoji = spl[1]
    score = spl[2]

    ret = '{\n  "CLIP": [\n'
    ret = ret + '    { "keyword": "' + key + '" },\n'
    ret = ret + '    { "emoji": "' + emoji + '" },\n'
    ret = ret + '    { "confidence": "' + score + '" }\n'
    ret = ret + '  ]\n}'

    return ret

def process_image(image_path):
    similarity_scores = compute_similarity(image_path, model, label_tensor)
 
    top_label_idx = similarity_scores[0].argmax()
    top_label = labels[top_label_idx]
    top_score = similarity_scores[0][top_label_idx].item()

    #s_idx = top_label_idx - 1
    #second_label = labels[s_idx]
    #second_score = ss[s_idx].item()

    print(f"{{\n  label: \"{top_label}\",\n  score: {top_score:.2f}\n}}")
    #print(f"{{\n  label: \"{second_label}\",\n  score: {second_score:.2f}\n}}")

    os.remove(image_path) # delete image
    
    emojis = lookup_emoji(top_label, top_score)
    
    ret = buildJSON(emojis)

    #return emojis
    return ret
    #await message.channel.send(top_label)


# create folder for uploaded data
FOLDER = './'
#os.makedirs(FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'GET':
        url = request.args.get('url')
        path = request.args.get('path')

        if url:
            print (url)
            fn =  uuid.uuid4().hex + ".jpg"

            response = requests.get(url)
            with open(fn, mode="wb") as file:
                file.write(response.content)

            emojis = process_image(fn)

            return emojis
        elif path:
            emojis = process_image(path)

            return emojis
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
        emojis = []
        for field, data in request.files.items():
            fn = uuid.uuid4().hex + ".jpg"
            print('field:', field)
            print('filename:', data.filename)
            print('UUID:', fn)
            if data.filename:
                data.save(os.path.join(FOLDER, fn)) #data.filename
                emojis = process_image(fn)

        #return "OK"
        return emojis

if __name__ == '__main__':
    # Debug/Development
    app.run(use_reloader=False, debug=True, host="0.0.0.0", port="7777")
    # Production
    #http_server = WSGIServer(('', 7777), app)
    #http_server.serve_forever()

#if __name__ == '__main__':
#    app.run(port=7777)