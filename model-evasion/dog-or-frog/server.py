#!/usr/bin/env python3

from flask import Flask, render_template, request
import io
from imagehash import phash
from keras_applications import imagenet_utils
from keras.applications.mobilenet import decode_predictions, preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')
# Probably not needed for production, but I have GPU support enabled on my version
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

imagenet_utils.CLASS_INDEX_PATH = '/challenge/static/imagenet_class_index.json'
TREE_FROG_ID = 31
TREE_FROG_STR = "tree_frog"
THRESHOLD = 0.95
PHASH_TRESH = 2
IMAGE_DIMS = (224, 224)
FLAG = open('/flag').read().strip()

app = Flask(__name__)
model = None
base_img = None

# Heavily taken from https://blog.keras.io/category/tutorials.html

def import_model():
    global model
    model = load_model("/challenge/static/model.h5")
    global base_img
    base_img = Image.open("/challenge/static/img/trixi.png").resize(IMAGE_DIMS)

def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(IMAGE_DIMS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image


def get_predictions(image):
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


def hash_hamming_distance(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))


@app.route("/predict", methods=["POST"])
def predict():
    global model

    # Due to some wonkiness with how Flask is loaded and the Keras backend,
    # the model has issues when it's loaded elsewhere.
    if model is None:
        import_model()

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("frog"):

            frog_img = Image.open(io.BytesIO(request.files["frog"].read()))

            frog_dist = hash_hamming_distance(phash(frog_img), phash(base_img))
            frog_mat = prepare_image(frog_img)

            # read the image in PIL format
            frog_label, frog_conf, top_preds = get_predictions(frog_mat)

            res = {}
            res["is_frog"] = TREE_FROG_STR in frog_label
            res["frog_conf"] = frog_conf
            res["frog_cat"] = frog_label
            res["frog_img_sim"] = frog_dist
            res["top_5"] = top_preds

            if TREE_FROG_STR in frog_label and frog_conf >= THRESHOLD and frog_dist <= PHASH_TRESH:
                return render_template("win.html", flag=FLAG)
            else:
                return render_template("results.html", results=res)

    return "Image processing fail"

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

if __name__ == '__main__':
    host, port = 'challenge.localhost', 80
    app.secret_key = os.urandom(8)
    app.config['SERVER_NAME'] = f'{host}:{port}'
    app.run(host, port)
