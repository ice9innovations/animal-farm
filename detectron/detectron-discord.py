# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

import discord
import json
import requests
import os
from dotenv import load_dotenv
import uuid

from urllib.parse import urlparse, parse_qs

from PIL import Image

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
CHANNELS = os.getenv('DISCORD_CHANNEL')
CHANNELS = CHANNELS.split(",")

HOST =  os.getenv('MYSQL_HOST')
USER =  os.getenv('MYSQL_USERNAME')
PW =  os.getenv('MYSQL_PASSWORD')
DB = os.getenv('MYSQL_DB')

CONFIG_FILE = "./mask_rcnn_R_50_FPN_3x.yaml"

#OPTS = {}
#OPTS[0] = "MODEL.WEIGHTS"
#OPTS["MODEL.WEIGHTS"] = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(CONFIG_FILE)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        #default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        #default="../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",

        default="mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
    


mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg()

demo = VisualizationDemo(cfg)

async def classify(image_path, message, image_url):
    print("IMAGE PATH: ")
    print(image_path)

    if image_path:
        if len(image_path) == 1:
            image_path = glob.glob(os.path.expanduser(image_path))
            assert image_path, "The input path(s) was not found"
        
        #for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                image_path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        #edit

        cc_file = open('coco_classes.txt', 'r')
        cc_classes = cc_file.read().splitlines()
        #print(cc_classes)

        #print(predictions)
        print("Predictions: ")
        instance = predictions.get("instances")
        sem_seg = predictions.get("proposals")
        clss = instance.get("pred_classes")
        confs = instance.get("scores")

        cnt = 0
        preds = []

        for cls in clss:
            tmp = int(cls.item())
            coco_object = cc_classes[tmp + 1]

            conf_cnt = 0
            conf_interval = None
            for conf in confs:
                if (conf_cnt == cnt):
                    conf_interval = conf.item()
                    #print(conf_interval)
                conf_cnt = conf_cnt + 1

            pred = coco_object + ":" + str(conf_interval)
            preds.append(pred)

            conf_interval = None #reset
            cnt = cnt + 1

        os.remove(image_path)
        await emoji(message, preds, image_url)

            #await reply(message, ret, image_url)
        #print(clss)
        #print(conf)

        #vals = tmp.get("scores")
        #print(vals)


        #if args.output:
        #    if os.path.isdir(args.output):
        #        assert os.path.isdir(args.output), args.output
        #        out_filename = os.path.join(args.output, os.path.basename(path))
        #    else:
        #        assert len(image_path) == 1, "Please specify a directory with args.output"
        #        out_filename = args.output

        #    visualized_output.save(out_filename)

        #else:
        #    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        #    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
        #    if cv2.waitKey(0) == 27:
        #        break  # esc to quit


            
    elif args.webcam:
        assert image_path is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()







intents = discord.Intents.default()
intents.message_content = True

#client = discord.Client(intents=discord.Intents.default())
client = discord.Client(intents=intents)

async def tnail(img, message, image_url):
    try:
        tn_filename = uuid.uuid4().hex + ".jpg"

        image = Image.open(img)
        image.thumbnail((160,160))
        image.save(tn_filename)
        #image1 = Image.open('tn.png')
        #image1.show()

        os.remove(img) # delete the original image

        await classify(tn_filename, message, image_url)

    except IOError:
        pass
   
async def download_image(image_file, message, image_url):
    print(image_file)

    #raw_image = Image.open(requests.get(image_file, stream=True).raw).convert('RGB')
    #raw_image = Image.open(open(img_file, "rb")).convert('RGB')
    #w,h = raw_image.size
    #transform = transforms.Compose([
    #    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #    ])
    #image = transform(raw_image).unsqueeze(0).to(device)
    
    response = requests.get(image_file)

    dl_filename = uuid.uuid4().hex + ".jpg"
    open(dl_filename, "wb").write(response.content)
    await tnail(dl_filename, message, image_url)

    return dl_filename #image


async def emoji(msg, preds, image_url):
    emojis = []
    #print(preds)
    #tags = preds.split(",")

    max = 3
    count = 0
    threshold = .33

    for pred in preds:
        if (pred):
            if (count <= max):
                count = count + 1

                print(pred)
                #current_emoji = 
                aPred = pred.split(":")
                
                tag = aPred[0]
                strength = aPred[1]

                print("Tag: " + str(tag))
                print("Strength: " + str(strength))
                await lookup_emoji(msg, tag, image_url)
                
                #if (strength > threshold):
                    #await lookup_emoji(msg, tag, image_url)
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

async def update_database(msg, image_url, reaction):
    #mycursor = mydb.cursor()

    sql = "INSERT INTO discord (server, channel, user, bot, image, reaction) VALUES (%s, %s, %s, %s, %s, %s)"
    val = (msg.guild.id, msg.channel.id, msg.author.id, "inception_v3", image_url, reaction)
    #mycursor.execute(sql, val)

    #mydb.commit()

async def reply(msg, body):
    text = "Null"
    if (body):
        emu = await emoji(msg, body)
        print(emu)
        text = body


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
            await download_image(url, message, url)

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

