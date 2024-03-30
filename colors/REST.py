#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
from flask import Flask, request

import json
import requests
import os
from dotenv import load_dotenv
import uuid
import re

from color_names import *
from haishoku.haishoku import Haishoku

#from urllib.parse import urlparse, parse_qs
from PIL import Image

color_emojis = {
    "red": "🟥",
    "green": "🟩",
    "blue": "🟦",
    "yellow": "🟨",
    "orange": "🟧",
    "purple": "🟪",
    "brown": "🟫",
    "black": "⬛",
    "white": "⬜"
}

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

#mydb = mysql.connector.connect(
#  host=HOST,
#  user=USER,
#  password=PW,
#  database=DB,
#  charset='utf8mb4'
#)

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def hex2rgb(hexcode):
    return tuple(map(ord,hexcode[1:].decode('hex')))

def get_color_by_name(name, style):
    style = str(style).lower()

    if style == "copic":
        items = copic.items()
    elif style == "prismacolor":
        items = prismacolor.items()

    for color, value in items:
        if name == color:
            return value 
        
def get_color_name(rgb, style):
    global copic
    style = str(style).lower()

    if style == "copic":
        items = copic.items()
    elif style == "prismacolor":
        items = prismacolor.items()

    min_distance = float("inf")
    closest_color = None
    for color, value in items:
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def get_color_name_for_emojis(rgb):
    global colors

    min_distance = float("inf")
    closest_color = None
    for color, value in colors_for_emojis.items():
        distance = sum([(i - j) ** 2 for i, j in zip(rgb, value)])
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def get_emoji_by_name(color_name):
    for color, emoji in color_emojis.items():
        if color == color_name:
            return emoji

def format_copic(name):
    copic = []
    copic_name = re.sub('[\(\[].*?[\)\]]', '', str(name)).strip()
    copic_code = name.replace(copic_name,"").replace("(","").replace(")","").strip()

    copic.append(copic_name)
    copic.append(copic_code)
    
    return copic

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def color_names(cnames, style):
    pal_json = '  {\n    "palette": [\n'

    pal_json = pal_json + '      {\n        "' + style + '": \n          ['
    #chex = []
    for name in cnames:
        n = get_color_by_name(name, style)
        #pretty name
        if (n):
            cname = format_copic(name)
            cstr = (cname[0] + " (" + cname[1]) + ")"
            pal_json = pal_json + '\n            {\n              "color": "' + cstr + '",\n'

            #print (copic_code)

            #print(n)
            h = rgb2hex(n[0],n[1],n[2])
            pal_json = pal_json + '              "rgb": [' + str(n[0]) + ',' + str(n[1]) + ',' + str(n[2]) + '],\n'
            pal_json = pal_json + '              "hex": "' + h + '"\n            },'

            #chex.append(h)
    #print (chex)
    
    #strip last comma
    pal_json = pal_json.rstrip(',')

    #close json
    pal_json = pal_json + "\n        ]\n      }\n    ]\n  }"
    #print (pal_json)

    return pal_json

def emoji(dominant):
    emo_json = '  {\n    "emo": {\n'
    # get dominant color name for emojis
    color_name_emoji = get_color_name_for_emojis(dominant)
    #print(color_name_emoji)
    emo_json = emo_json + '      "color": "' + color_name_emoji + '",\n'

    #get emoji from color name 
    color_emoji = get_emoji_by_name(color_name_emoji)
    #print (color_emoji)
    
    emo_json = emo_json + '      "emoji": "' + color_emoji + '"\n    }\n  }'
    #print (emo_json)

    return emo_json

def dominant(dl_filename):
    dom_json = '  {\n    "dominant": {\n'

    dominant = Haishoku.getDominant(dl_filename)
    dr = dominant[0]
    dg = dominant[1]
    db = dominant[2]
    dhex = rgb2hex(dr,dg,db)
    #print(dhex)

    #dominant color name copic
    dominant_copic = get_color_name(dominant, "copic")
    copic_d = format_copic(dominant_copic)
    copic_str = (copic_d[0] + " (" + copic_d[1]) + ")"

    #dominant color name prismacolor
    dominant_prisma = get_color_name(dominant, "prismacolor")
    prisma_d = format_copic(dominant_prisma)
    prisma_str = (prisma_d[0] + " (" + prisma_d[1]) + ")"

    #build emoji json based on dominant color
    emoji_json = emoji(dominant)
    #print (emoji_json)

    #build json for dominant color
    dom_json = dom_json + '      "copic": "' + copic_str + '",\n'
    dom_json = dom_json + '      "prismacolor": "' + prisma_str + '",\n'
    dom_json = dom_json + '      "rgb": [' + str(dr) + ',' + str(dg) + ',' + str(db) + '],\n'
    dom_json = dom_json + '      "hex": "' + dhex + '"\n    }\n  }'
    #print (dom_json)

    #build json and return
    consolidated_json = emoji_json + ",\n" + dom_json
    #print(consolidated_json)

    return consolidated_json

def palette(dl_filename):
    #get RGB color palette from Haishoku
    palette = Haishoku.getPalette(dl_filename)
    hexes = []
    cnames_copic = []
    cnames_prisma = []

    for p in palette:
        clr = p[1]
        #print(clr)
        cnames_copic.append(get_color_name(clr, "copic"))
        cnames_prisma.append(get_color_name(clr, "prismacolor"))

        r = clr[0]
        g = clr[1]
        b = clr[2]

        hex = rgb2hex(r,g,b)
        #print(hex)
        hexes.append(hex)

    cnames_copic = unique(cnames_copic)
    cnames_prisma = unique(cnames_prisma)

    #print (cnames_copic)
    #print (cnames_prisma)

    cnc = color_names(cnames_copic, "copic")
    cnp = color_names(cnames_prisma, "prismacolor")
    #print (cnc)
    #print (cnp)

    consolidated = cnc + ",\n" + cnp
    return consolidated

def classify(image_file):
    #print(image_file)
    ret = {}

    dom = dominant(image_file)
    #emo = emoji(dom)
    pal = palette(image_file)
    
    print (dom)
    print (pal)
    consolidated = '{\n  "colors": [\n' + dom + ',\n' + pal + '\n]}'

    os.remove(image_file)
    return consolidated #image

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
            fn = uuid.uuid4().hex + ".jpg"

            response = requests.get(url)
            with open(fn, mode="wb") as file:
                file.write(response.content)

            colors = classify(fn)
            print(colors)

            return colors
        elif path:
            colors = classify(path)
            print(colors)

            return colors
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
                colors = classify(fn)
                print(colors)

        #return "OK"
        return colors

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    #IP = "127.0.0.1"
    app.run(use_reloader=False,debug=True, host="0.0.0.0", port=7770)


