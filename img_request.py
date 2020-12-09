import requests
import argparse

api_url = 'http://127.0.0.1:5000/api'


def get_args():
    parser = argparse.ArgumentParser('api server request')
    parser.add_argument('--image_path', type=str, default='temp/0abn.jpg', help='image path to test')
    args = parser.parse_args()
    return args


def predict_result(image_path):
    image = open(image_path, 'rb').read()
    payload = {'image': image}
    r = requests.post(api_url, files=payload).json()


if __name__ == '__main__':
    opt = get_args()
    predict_result(opt.image_path)
