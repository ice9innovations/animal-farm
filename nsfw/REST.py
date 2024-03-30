import json
import requests
import os
from dotenv import load_dotenv
import uuid

from typing import List
from typing import Any, Tuple

import tensorflow as tf
from absl import logging as absl_logging
  
# TFA is EOL in May, so I copied the necessary functions into this file
# and removed the dependency

#from private_detector.utils.preprocess import preprocess_for_evaluation

from flask import Flask, request

tf.get_logger().setLevel('ERROR')
absl_logging.set_verbosity(absl_logging.ERROR)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

FOLDER = './'

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

PROBABILITY = 35

def preprocess_for_evaluation(image: tf.Tensor,
                              image_size: int,
                              dtype: tf.dtypes.DType) -> tf.Tensor:
    """
    Preprocess image for evaluation

    Parameters
    ----------
    image : tf.Tensor
        Image to be preprocessed
    image_size : int
        Height/Width of image to be resized to
    dtype : tf.dtypes.DType
        Dtype of image to be used

    Returns
    -------
    image : tf.Tensor
        Image ready for evaluation
    """
    image = pad_resize_image(
        image,
        [image_size, image_size]
    )

    image = tf.cast(image, dtype)

    image -= 128
    image /= 128

    return image


def pad_resize_image(image: tf.Tensor,
                     dims: Tuple[int, int]) -> tf.Tensor:
    """
    Resize image with padding

    Parameters
    ----------
    image : tf.Tensor
        Image to resize
    dims : Tuple[int, int]
        Dimensions of resized image

    Returns
    -------
    image : tf.Tensor
        Resized image
    """
    image = tf.image.resize(
        image,
        dims,
        preserve_aspect_ratio=True
    )

    shape = tf.shape(image)

    sxd = dims[1] - shape[1]
    syd = dims[0] - shape[0]

    sx = tf.cast(
        sxd / 2,
        dtype=tf.int32
    )
    sy = tf.cast(
        syd / 2,
        dtype=tf.int32
    )

    paddings = tf.convert_to_tensor([
        [sy, syd - sy],
        [sx, sxd - sx],
        [0, 0]
    ])

    image = tf.pad(
        image,
        paddings,
        mode='CONSTANT',
        constant_values=128
    )

    return image

def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector

    Parameters
    ----------
    filename : str
        Filename of image

    Returns
    -------
    image : tf.Tensor
        Image ready for inference
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(
        image,
        480,
        tf.float16
    )

    image = tf.reshape(image, -1)

    return image


#model: str , 
model = "./saved_model/"
model = tf.saved_model.load(model)
def inference(image_path) -> None:
    global model
    global PROBABILITY
    
    """
    Get predictions with a Private Detector model

    Parameters
    ----------
    model : str
        Path to saved model
    image_paths : List[str]
        Path(s) to image to be predicted on
    """

    #for image_path in image_paths:
    print (image_path)

    try:
        error = False
        image = read_image(FOLDER + image_path)
        preds = model([image])
    except:
        error = True

    if not error:
        #print(f'Probability: {100 * tf.get_static_value(preds[0])[0]:.2f}% - {image_path}')
        prob = round(tf.get_static_value(preds[0])[0],3)
    else:
        prob = -1

    ret = '{\n"nsfw": [\n  {"confidence": ' +  str(prob) + '},'

    emoji = ""    
    if prob > PROBABILITY:
        emoji = "🚫"

    ret = ret + '\n  {"emoji": "' + emoji + '"}\n  ]\n}'
    os.remove(os.path.join(FOLDER, image_path))

    return ret


# create folder for uploaded data
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

            nsfw = inference(fn)

            print(nsfw)

            return nsfw
        elif path:
            nsfw = inference(path)
            print(nsfw)

            return nsfw
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
                nsfw = inference(fn)
                print(nsfw)

        #return "OK"
        return nsfw

if __name__ == '__main__':
    # Debug/Development
    IP = "0.0.0.0"
    IP = "127.0.0.1"
    app.run(use_reloader=False,debug=True, host="0.0.0.0", port=7774)


