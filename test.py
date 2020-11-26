import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from parameters import *
import torch as t
import torch.nn.functional as F
from dataset import StrtoLabel, LabeltoStr
from model import *


nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

if __name__ == '__main__':
    img_path = './userTest/0Up4.jpg'
    transform = T.Compose([
            T.Resize((ImageHeight, ImageWidth)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    label = img_path.split("/")[-1].split(".")[0]
    img = Image.open(img_path)
    img = transform(img)
    img = t.unsqueeze(img, 0)

    net = ResNet(ResidualBlock)
    net.eval()
    weight_path = 'weights/captcha/resnet_best.pth'
    net.load_state_dict(t.load(weight_path))
    y1, y2, y3, y4 = net(img)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                     y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
    y = t.cat((y1, y2, y3, y4), dim=1)
    # print(x,label,y)
    str = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])

    if str == label:
        predict = True
    else:
        predict = False
    print('the output is {}, the label is {}, predict is {}'.format(str, label, predict))