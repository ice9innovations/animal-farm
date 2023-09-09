#import mysql.connector
import re

import os
import os.path

import requests
import json
import discord
from dotenv import load_dotenv
import uuid

from PIL import Image

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import argparse
import time
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

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

print(device)

parser = argparse.ArgumentParser()
#parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default="", help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=160, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.19, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_false', help='don`t trace model')
opt = parser.parse_args()

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

async def emoji(msg, preds, discord_image):
    emojis = []
    tags = preds.split(",")
    count = 1
    for tag in tags:
        if tag and (count <= max):
            #current_emoji = 
            #print("COUNT: " + str(count))
            count = count + 1
            await lookup_emoji(msg, tag, discord_image)
            #print(current_emoji)

async def lookup_emoji(msg, tag, discord_image):
    f = open('./emojis.json')
    data = json.load(f)

    for d in data:
        for key, value in d.items():
            #print(key, value)
            aTags = tag.split(", ")
        
            if len(aTags) > 0:
                for aTag in aTags:
                    aTag = aTag.strip()

                    clean_tags = aTag.split(" ")

                    cnt = 0
                    for clean_tag in clean_tags:
                        #if clean_tags[1] and len(clean_tags) > 0: #why???
                        if cnt == 1:
                            clean_tag = clean_tags[1]
                            clean_tag = singularize(clean_tag.replace(",","").strip()) # drop extra comma

                            if (value == clean_tag or key == clean_tag):
                                #print("TAG: " + clean_tag)
                                print(key, value)

                                if value:
                                    await update_database(msg, discord_image, value)
                                    await msg.add_reaction(value)

                        cnt = cnt + 1

                #return value

def download_image(image_file,image_size,device):
    print(image_file)

    raw_image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    #raw_image = Image.open(open(img_file, "rb")).convert('RGB')

    w,h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(raw_image).unsqueeze(0).to(device)
    
    dl_filename = uuid.uuid4().hex + ".jpg"

    response = requests.get(image_file)
    open(dl_filename, "wb").write(response.content)

    return dl_filename #image

async def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "YOLO", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

async def detect(discord_image, message, device):
    #print("Detecting: " + discord_image)
    
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

    opt.img = download_image(discord_image, 160, device)
    
    #print("OPT IMG: " + opt.img)
    img = opt.img
    save_img = False

    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    #device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    list_all = ""

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, opt.img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        list_all += line
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}')

            if os.path.isfile(path):
                os.remove(path)

            await reply(message, f'{s}', discord_image)
            #c = "{ caption: " + "'" + caption[0] + "'}"

            #print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    #print(f'Done. ({time.time() - t0:.3f}s)')

intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)

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
    #print("Message received.")

    #if message.author.bot: # or message.channel.name != CHANNEL:
    #    return

    #print(`"CHANNELS")
    channel_access = False
    for CHANNEL in CHANNELS:
        if CHANNEL == message.channel.name:
            channel_access = True
            
    if channel_access:
        #if message.channel.name in CHANNEL:
        #if not message.content or len(message.attachments < 1):
            # You can use message.channel.send() if you don't want to mention user
            #await message.reply("Message should contain an image")
            #return
        #print ("Scanning attachments")

        for attachment in message.attachments:
            url = attachment.url
            response = "URL: " + url
            with torch.no_grad():
                if opt.update:  # update all models (to fix SourceChangeWarning)
                    for opt.weights in ['yolov7.pt']:
                        await detect(url, message, device)
                        strip_optimizer(opt.weights)
                else:
                    await detect(url, message, device)

  #      if len(message.attachments) < 1:
            #print(len(message.attachments))
   #         url = message.attachments[0].url
    else:
        print("NO CHANNEL ACCESS")


async def reply(msg, body, discord_image):
    text = "Null"
    if (body):
        emu = await emoji(msg, body, discord_image)
        print(emu)
        text = body

    #await msg.channel.send(text)

#async def on_message(message):

client.run(TOKEN)
