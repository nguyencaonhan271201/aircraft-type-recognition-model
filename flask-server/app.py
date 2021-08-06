from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import torch
import PIL
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import activations as Ac
from tensorflow.keras.callbacks import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from airplane_detection import predict_image
from detect import run

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
model = None

# home page
@app.route('/')
def home():
   return render_template('index.html', image_file_name = "vna.jpg", result1 = "",
   result2 = "", result3 = "", result4 = "", result5 = "")

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        x1, y1, x2, y2, result = detect(full_name)

    return render_template('index.html', image_file_name = file.filename, x1 = x1, y1 = y1, x2 = x2, y2 = y2,
    result1 = f"{result[0][0]} ({result[0][1]}%)", result2 = f"{result[1][0]} ({result[1][1]}%)", 
    result3 = f"{result[2][0]} ({result[2][1]}%)", result4 = f"{result[3][0]} ({result[3][1]}%)", 
    result5 = f"{result[4][0]} ({result[4][1]}%)")


def get_bounding_boxes_coordinates(image_path):
    # Inference
    #results = model(image_path)
    results = run(source=image_path)
    largest_area = 0
    count = 0
    for box in results:
        if (box[5] == 4):
            area = abs(box[2] - box[0]) * abs(box[3] - box[1])
            if area > largest_area:
                largest_area = area
                best_box = box
            count += 1
    if (count == 0):
        #Not find any boxes of class aeroplane
        image = PIL.Image.open(image_path) 
        width, height = image.size
        x1 = 0
        y1 = 0
        x2 = width
        y2 = height
    else:
        x1 = float(best_box[0])
        y1 = float(best_box[1])
        x2 = float(best_box[2])
        y2 = float(best_box[3])
        
    return (x1, y1, x2, y2)

def detect(img_path):
    x1, y1, x2, y2 = get_bounding_boxes_coordinates(img_path)
    labels = predict_image(img_path, x1, y1, x2, y2)
    print(x1, y1, x2, y2)
    return x1, y1, x2, y2, labels
    
if __name__ == "__main__":
    '''
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True
    ).autoshape()  # force_reload = recache latest code
    model.eval()
    '''
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)