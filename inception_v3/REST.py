#https://discord.com/oauth2/authorize?client_id=1121941169476222987&permissions=3136&scope=bot%20applications.commands%20messages.read
#INSERT INTO discord (server,channel,user,bot,image,reaction) VALUES ('test','test','test','test','test','test');

#import mysql.connector
# import the necessary packages
from flask import Flask, request

import json
import requests
import os
from dotenv import load_dotenv
import uuid

from urllib.parse import urlparse, parse_qs

from PIL import Image
import inception

PRIVATE = False # change this for security to limit API requests to only local access
API_PORT = os.getenv('API_PORT')

load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

model = inception.Inception()

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

def tnail(img):
    tn_filename = ""

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

    return tn_filename

def emoji(preds):
    emojis = ""
    #print(preds)
    #tags = preds.split(",")

    max = 3
    count = 0
    threshold = .15

    for pred in preds:
        if (pred):
            if (count <= max):
                count = count + 1

                print(pred)
                #current_emoji = 
                aPred = pred.split(":")
                
                tag = aPred[0]
                strength = round(float(aPred[1]),3)

                print("Strength: " + str(strength))
                
                if (strength > threshold):
                    emo = lookup_emoji(tag)
                    if (emo):
                        tmp = '{"tag": "' + tag + '", "confidence": "' + str(strength) + '", "emoji": "' + emo + '"},' 
                        emojis = emojis + tmp
                    #print(current_emoji)

    emojis = "[" + emojis.rstrip(",") + "]"
    return emojis

def lookup_emoji(tag):
    f = open('./emojis.json')
    data = json.load(f)

    print("Tag: " + tag )
    for d in data:
        for key, value in d.items():
            #print(key, value)
            if (value == tag or key == tag):
                print(key, value)
                #if (key):
                    #await update_database(msg, image_url, value)
                    #await msg.add_reaction(value)

                return value

def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

def classify(image_path):
    global model
    # Display the image.
    #display(Image(image_path))

    tn = tnail(image_path)

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=tn)

    # Print the scores and names for the top-10 predictions.
    ret = model.print_scores(pred=pred, k=10, only_first_name=True)
    #ret = json.dumps(str(ret))
    #print(ret)
    os.remove(tn)

    return ret
    #ret = pred
    #www.send_response(200)
    #www.send_header("Content-type", "text/txt")
    #www.end_headers()

    #www.wfile.write(ret.encode()) #.encode()


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
                print(emojis)

                emo = emoji(emojis)
                print(emojis)

        #return "OK"
        return emo

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    if PRIVATE:
        IP = "127.0.0.1"

    app.run(use_reloader=False,debug=True, host=IP, port=7779)


