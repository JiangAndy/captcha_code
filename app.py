from torchvision import transforms as T
from parameters import *
import torch as t
import torch.nn.functional as F
from dataset import StrtoLabel, LabeltoStr
from model import *
from generate_captcha import *
import base64
import json
import requests
from test import predict, img2base64, base642img
from flask import Flask, request, jsonify
import io, os
from PIL import Image
import cv2
import numpy as np

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

app = Flask(__name__)
temp_dir = './temp'
weight_path = 'weights/captcha/resnet_best.pth'
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


@app.route('/api/', methods=["POST"])
def inference():
    if request.method == 'POST':
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_path = os.path.join(temp_dir, 'temp.jpg')
            cv2.imwrite(img_path, img_arr)
            pred = predict(img_path, weight_path)
            print(pred)
            results = {"results": pred}
            return jsonify(results)


if __name__ == '__main__':
    app.run(host='127.0.0.1')
