import os
from PIL import Image
import numpy as np
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


nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']


def generate_img(temp_dir):
    '''
    generate a random image
    :param temp_dir: the dir for save temp image
    :return: the ground truth of the temp image
    '''
    font_sizes = [x for x in range(40, 45)]
    imc = ImageCaptcha(get_width(), get_height(), font_sizes=font_sizes)
    name = get_string()
    image = imc.generate_image(name)
    image.save(os.path.join(temp_dir,  'temp.jpg'))
    return name


def predict(img_path, weight_path):
    '''
    predict the image
    :param img_path: image path to predict
    :param weight_path: model weights path
    :return: the results of predict
    '''
    transform = T.Compose([
        T.Resize((ImageHeight, ImageWidth)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(img_path)
    img = transform(img)
    img = t.unsqueeze(img, 0)

    net = ResNet(ResidualBlock)
    net.eval()
    net.load_state_dict(t.load(weight_path))
    y1, y2, y3, y4 = net(img)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                     y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = t.cat((y1, y2, y3, y4), dim=1)
    pred = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])

    return pred


def img2base64(img_path):
    '''
    convert image to base64
    :param img_path: image path
    :return: base64 data of image
    '''
    f = open(img_path, 'rb')
    bf = base64.b64encode(f.read())
    f.close()
    return bf


def base642img(bf):
    '''
    convert base64 data to image
    :param bf: base64 code
    :return: image
    '''
    img_data = base64.b64decode(bf)
    file = open('./temp/temp.jpg', 'wb')
    file.write(img_data)
    file.close()


# todo finish the api
def recognize_api(temp_dir, url):
    '''
    recognize api
    :param temp_dir: the dir for save temp image
    :param url:
    :return: results
    '''
    data = {}
    name = generate_img(temp_dir)
    img_path = os.path.join(temp_dir, 'temp.jpg')
    img_base64 = img2base64(img_path)
    pred = predict(img_path, weight_path)
    data['image_base64'] = str(img_base64, 'utf-8')
    data['pred'] = pred
    headers = {'Content-Type': 'application/json'}
    res = requests.post(url=url, headers=headers, data=json.dump(data))
    res = res.json()
    return res['string']


if __name__ == '__main__':
    temp_dir = './temp'
    weight_path = 'weights/captcha/resnet_best.pth'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    name = generate_img(temp_dir)
    img_path = os.path.join(temp_dir, 'temp.jpg')
    img_base64 = img2base64(img_path)

    pred = predict(img_path, weight_path)
    right = pred == name
    print('predict is {}, ans is {}'.format(pred, right))