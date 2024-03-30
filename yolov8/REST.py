#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
# import the necessary packages
from flask import Flask, request

from ultralytics import YOLO

import json
import requests
import os
from dotenv import load_dotenv
import uuid

from urllib.parse import urlparse, parse_qs
from PIL import Image

import argparse
import time
from pathlib import Path

import re
import torch

PRIVATE = False # change this for security to limit API requests to only local access
API_PORT = os.getenv('API_PORT')

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

#HOST =  os.getenv('MYSQL_HOST')
#USER =  os.getenv('MYSQL_USERNAME')
#PW =  os.getenv('MYSQL_PASSWORD')
#DB = os.getenv('MYSQL_DB')

#mydb = mysql.connector.connect(
#  host=HOST,
#  user=USER,
#  password=PW,
#  database=DB,
#  charset='utf8mb4'
#)

image_size = 160
max = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device("mps")


# copied and pasted directly from
# https://gist.github.com/gaulinmp/7107e3bac5ea94af6c9d

_singular_rules = [
    (r'(?i)(.)ae$', '\\1a'),
    (r'(?i)(.)itis$', '\\1itis'),
    (r'(?i)(.)eaux$', '\\1eau'),
    (r'(?i)(quiz)zes$', '\\1'),
    (r'(?i)(matr)ices$', '\\1ix'),
    (r'(?i)(ap|vert|ind)ices$', '\\1ex'),
    (r'(?i)^(ox)en', '\\1'),
    (r'(?i)(alias|status)es$', '\\1'),
    (r'(?i)([octop|vir])i$',  '\\1us'),
    (r'(?i)(cris|ax|test)es$', '\\1is'),
    (r'(?i)(shoe)s$', '\\1'),
    (r'(?i)(o)es$', '\\1'),
    (r'(?i)(bus)es$', '\\1'),
    (r'(?i)([m|l])ice$', '\\1ouse'),
    (r'(?i)(x|ch|ss|sh)es$', '\\1'),
    (r'(?i)(m)ovies$', '\\1ovie'),
    (r'(?i)(.)ombies$', '\\1ombie'),
    (r'(?i)(s)eries$', '\\1eries'),
    (r'(?i)([^aeiouy]|qu)ies$', '\\1y'),
    # -f, -fe sometimes take -ves in the plural
    # (e.g., lives, wolves).
    (r"([aeo]l)ves$", "\\1f"),
    (r"([^d]ea)ves$", "\\1f"),
    (r"arves$", "arf"),
    (r"erves$", "erve"),
    (r"([nlw]i)ves$", "\\1fe"),
    (r'(?i)([lr])ves$', '\\1f'),
    (r"([aeo])ves$", "\\1ve"),
    (r'(?i)(sive)s$', '\\1'),
    (r'(?i)(tive)s$', '\\1'),
    (r'(?i)(hive)s$', '\\1'),
    (r'(?i)([^f])ves$', '\\1fe'),
    # -ses suffixes.
    (r'(?i)(^analy)ses$', '\\1sis'),
    (r'(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$',
     '\\1\\2sis'),
    (r'(?i)(.)opses$', '\\1opsis'),
    (r'(?i)(.)yses$', '\\1ysis'),
    (r'(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'),
    (r'(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$',
     '\\1ose'),
    (r'(?i)(.)oses$', '\\1osis'),
    # -a
    (r'(?i)([ti])a$', '\\1um'),
    (r'(?i)(n)ews$', '\\1ews'),
    (r'(?i)([^s])s$', '\\1'),  # don't make ss singularize to s.
]

# For performance, compile the regular expressions only once:
_singular_rules = [(re.compile(r[0]), r[1]) for r in _singular_rules]

_singular_uninflected = set((
    "bison", "debris", "headquarters", "pincers", "trout",
    "bream", "diabetes", "herpes", "pliers", "tuna",
    "breeches", "djinn", "high-jinks", "proceedings", "whiting",
    "britches", "eland", "homework", "rabies", "wildebeest"
    "carp", "elk", "innings", "salmon",
    "chassis", "flounder", "jackanapes", "scissors",
    "christmas", "gallows", "mackerel", "series",
    "clippers", "georgia", "measles", "shears",
    "cod", "graffiti", "mews", "species",
    "contretemps", "mumps", "swine",
    "corps", "news", "swiss",
    # Custom added from MD&A corpus
    "api", "mae", "sae", "basis", "india", "media",
))
_singular_uncountable = set((
    "advice", "equipment", "happiness", "luggage", "news", "software",
    "bread", "fruit", "information", "mathematics", "progress", "understanding",
    "butter", "furniture", "ketchup", "mayonnaise", "research", "water"
    "cheese", "garbage", "knowledge", "meat", "rice",
    "electricity", "gravel", "love", "mustard", "sand",
))
_singular_ie = set((
    "alergie", "cutie", "hoagie", "newbie", "softie", "veggie",
    "auntie", "doggie", "hottie", "nightie", "sortie", "weenie",
    "beanie", "eyrie", "indie", "oldie", "stoolie", "yuppie",
    "birdie", "freebie", "junkie", "^pie", "sweetie", "zombie"
    "bogie", "goonie", "laddie", "pixie", "techie",
    "bombie", "groupie", "laramie", "quickie", "^tie",
    "collie", "hankie", "lingerie", "reverie", "toughie",
    "cookie", "hippie", "meanie", "rookie", "valkyrie",
))
_singular_irregular = {
    "abuses": "abuse",
    "ads": "ad",
    "atlantes": "atlas",
    "atlases": "atlas",
    "analysis": "analysis",
    "axes": "axe",
    "beeves": "beef",
    "brethren": "brother",
    "children": "child",
    "children": "child",
    "corpora": "corpus",
    "corpuses": "corpus",
    "ephemerides": "ephemeris",
    "feet": "foot",
    "ganglia": "ganglion",
    "geese": "goose",
    "genera": "genus",
    "genii": "genie",
    "graffiti": "graffito",
    "helves": "helve",
    "kine": "cow",
    "leaves": "leaf",
    "loaves": "loaf",
    "men": "man",
    "mongooses": "mongoose",
    "monies": "money",
    "moves": "move",
    "mythoi": "mythos",
    "numena": "numen",
    "occipita": "occiput",
    "octopodes": "octopus",
    "opera": "opus",
    "opuses": "opus",
    "our": "my",
    "oxen": "ox",
    "penes": "penis",
    "penises": "penis",
    "people": "person",
    "sexes": "sex",
    "soliloquies": "soliloquy",
    "teeth": "tooth",
    "testes": "testis",
    "trilbys": "trilby",
    "turves": "turf",
    "zoa": "zoon",
}

_plural_prepositions = set((
    "about", "before", "during", "of", "till",
    "above", "behind", "except", "off", "to",
    "across", "below", "for", "on", "under",
    "after", "beneath", "from", "onto", "until",
    "among", "beside", "in", "out", "unto",
    "around", "besides", "into", "over", "upon",
    "at", "between", "near", "since", "with",
    "athwart", "betwixt", "beyond", "but", "by"
))

 
def singularize(word, custom={}):
    """Returns the singular of a given word."""
    if word in custom:
        return custom[word]
    # Recurse compound words (e.g. mothers-in-law).
    if "-" in word:
        w = word.split("-")
        if len(w) > 1 and w[1] in _plural_prepositions:
            return singularize(w[0], custom) + "-" + "-".join(w[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1], custom) + "'s"
    w = word.lower()
    for x in _singular_uninflected:
        if x.endswith(w):
            return word
    for x in _singular_uncountable:
        if x.endswith(w):
            return word
    for x in _singular_ie:
        if w.endswith(x + "s"):
            return w
    for x in _singular_irregular:
        if w.endswith(x):
            return re.sub('(?i)' + x + '$', _singular_irregular[x], word)
    for suffix, inflection in _singular_rules:
        m = suffix.search(word)
        g = m and m.groups() or []
        if m:
            for k in range(len(g)):
                if g[k] is None:
                    inflection = inflection.replace('\\' + str(k + 1), '')
            return suffix.sub(inflection, word)
    return word

def emoji(preds):
    emojis = []
    #print (preds)
    #tags = ''.join(preds).split(", ")
    count = 1
    emojis = []
    json_str = ""
    for tag in preds:
        print(tag)

        if tag != "" and tag != " ":
            aTag = tag.split("|")
            conf = round(float(aTag[1]), 3)
            print(conf)

            tag = aTag[0]

            #current_emoji = 
            #print("COUNT: " + str(count))
            count = count + 1
            emoji = lookup_emoji(tag)
            emoji = emoji.split(":")

            #if emoji != "" and emoji != " ":
            tag_str = '{"tag": "' + tag.strip() + '", "clean_tag": "' + emoji[0] + '", "confidence": ' + str(conf) + ',"emoji": "' + emoji[1] + '"},'
            json_str = json_str + tag_str
            #print(current_emoji)
    
    #json_str = '{"emojis": '
    #tmp = ""
    #for emo in emojis:
    #    tmp = tmp + emo + ","
    #json_str = json_str + '"' + tmp.rstrip(',') + '"}'
    json_str = "[" + json_str.rstrip(",") + "]" # remove last comma

    print (json_str)

    return json_str

def lookup_emoji(tag):
    f = open('./emojis.json')
    data = json.load(f)

    emoji = ""
    for d in data:
        for key, value in d.items():
            #print(key, value)
            aTags = tag.split(", ")

            if len(aTags) > 0:
                for aTag in aTags:
                    aTag = aTag.strip() #.replace(" ", "_")
                    #clean_tags = aTag.split(" ")

                    cnt = 0
                    clean_tag = aTag

                    #for clean_tag in clean_tags:
                    #if clean_tags[1] and len(clean_tags) > 0: #why???
                    clean_tag = re.sub(r'\b\d+\b ', '', clean_tag)
                    clean_tag = clean_tag.replace(" ","_")

                    #if cnt == 1:
                    #clean_tag = clean_tags[1]
                    clean_tag = singularize(clean_tag.replace(",","").strip()) # drop extra comma
                    clean_tag = clean_tag.replace(" ","_")
                    #print(clean_tag)

                    if (value == clean_tag or key == clean_tag):
                        #print("TAG: " + clean_tag)
                        print(key, value)

                        if value:
                            emoji = value
                            #await update_database(msg, discord_image, value)
                            #await msg.add_reaction(value)

                    #cnt = cnt + 1

    ret = clean_tag + ":" + emoji
    return ret

def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "YOLO", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()


def tnail(img):
    try:
        tn_filename = uuid.uuid4().hex + ".jpg"

        image = Image.open(img)
        image.thumbnail((160,160))
        image.save(tn_filename)
        #image1 = Image.open('tn.png')
        #image1.show()

        os.remove(img) # delete the original image


    except IOError:
        pass

#weights = opt.weights
#model = attempt_load(weights, map_location=device)  # load FP32 model
model = YOLO('yolov8l.pt')

def classify(discord_image):
    global model
    results = model.predict(discord_image,verbose=False)
    result = results[0]

    preds = []
    for box in result.boxes:

        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = result.names[box.cls[0].item()]
        conf = round(box.conf[0].item(), 2)

        print("Object type:", class_id)
        print("Coordinates:", cords)
        print("Probability:", conf)
        print("----")

        preds.append(class_id + "|" + str(round(conf,3)))
    
    path = "./" + discord_image
    try:
        os.remove(path)
    except:
        print("Cannot remove downloaded image")

    return preds

def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

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

            emojis = classify(fn)
            emo = emoji(emojis)
            print(emojis)
            #emo = emoji(emojis)

            return emo
        elif path:
            emojis = classify(path)
            emo = emoji(emojis)
            print(emojis)

            return emo
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
                emojis = classify(fn)
                emo = emoji(emojis)
                print(emojis)

        #return "OK"
        return emo

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    if PRIVATE:
        IP = "127.0.0.1"

    app.run(use_reloader=False,debug=True, host=IP, port=7776)


