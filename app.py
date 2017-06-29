from flask import Flask
from flask import jsonify, render_template
from matplotlib.pyplot import imsave

app = Flask(__name__)
import base64
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

# 偏移量
bias = -1

slide_captcha_url = 'http://api.shuibei.chxj.name/slide-captcha'


def predict():
    response = requests.get(slide_captcha_url)
    base64_image = response.json()['data']['dataUrl']
    base64_image_without_head = base64_image.replace('data:image/png;base64,', '')

    bytes_io = BytesIO(base64.b64decode(base64_image_without_head))
    img = np.array(Image.open(bytes_io).convert('RGB'))

    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 100, 200)

    operator = get_operator('shape.png')

    (x, y), _ = best_match(img_canny, operator)
    x = x + bias
    print('the position of x is', x)

    buffer = mark(img, x, y)

    return {'value': x, 'image': base64.b64encode(buffer.getbuffer()).decode()}


def get_operator(path, url=False, expand=False):
    if url:
        req = requests.get(path)
        arr = np.asarray(bytearray(req.content), dtype=np.uint8)
        shape = cv2.resize(cv2.imdecode(arr, -1), (69, 69))
    else:
        shape = cv2.resize(cv2.imread('shape.png'), (69, 69))

    shape_gray = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY)

    _, shape_binary = cv2.threshold(shape_gray, 127, 255, cv2.THRESH_BINARY)

    _, contours, hierarchy = cv2.findContours(shape_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    operator = np.zeros((69, 69))

    for point in contour:
        operator[point[0][0]][point[0][1]] = 1
        if expand:
            if point[0][0] > 0:
                operator[point[0][0] - 1][point[0][1]] = 1
            if point[0][0] < 68:
                operator[point[0][0] + 1][point[0][1]] = 1
            if point[0][1] > 0:
                operator[point[0][0]][point[0][1] - 1] = 1
            if point[0][1] < 68:
                operator[point[0][0]][point[0][1] + 1] = 1

    return operator


def best_match(image, operator):
    y_range, x_range = image.shape
    max_value, position = 0, (1, 1)

    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if y + 69 > 185 or x + 69 > 315:
                continue
            block = image[(y - 1):(y + 68), (x - 1):(x + 68)]
            value = (block * operator).sum()
            if value > max_value:
                max_value = value
                position = (x, y)

    return position, max_value


def mark(img, x, y):
    cv2.putText(img, 'O', (x - 15, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)

    buffer = BytesIO()
    imsave(buffer, img)
    return buffer


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch')
def fetch():
    return jsonify(predict())
