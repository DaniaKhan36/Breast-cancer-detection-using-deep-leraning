from flask import Flask, render_template, request 
from flask import redirect , url_for
import requests
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import pickle
import math
import tensorflow as tf
from keras.models import load_model
import cv2
# Import your image detection module
model = load_model('my_model2.hdf5')

app = Flask(__name__,static_url_path='/static',template_folder='templates')

IMGBB_API_KEY = 'a2e1a06dd5974afdae8e96f50cd9127e'

@app.route('/')
def home():
      return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict(): 
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # If the user does not select a file, browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Save the file to the uploads directory
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)

        # Upload the image to ImgBB using the ImgBB API
        image_url = upload_image_to_imgbb(file_path)
        #Prediction Part
        CATAGORIES=['bening','malignant','mormal']
        img=imread(image_url)
        img=cv2.resize(img,(224,224))
        img_array=np.array(img)
        img_array=img.reshape(1,224,224,3)

        a=model.predict(img_array)
        b=a.ravel().tolist()
        max_index = np.argmax(b)
        #indices=a.argmax()
        y_out=CATAGORIES[max_index]

        # Pass the image URL to the HTML template
        return render_template('index.html', image_url=image_url,predicted_text=y_out)
        #return render_template('index.html', image_url=image_url,predicted_text=y_out)

    return render_template('index.html')

def upload_image_to_imgbb(file_path):
    imgbb_url = 'https://api.imgbb.com/1/upload'
    with open(file_path, 'rb') as file:
        files = {'image': (file_path, file.read())}
        params = {'key':IMGBB_API_KEY}
        response = requests.post(imgbb_url, files=files, params=params)
        # Check the status code
        if response.status_code == 200:
            # Request was successful
            result = response.json()
            if result['success']:
                return result['data']['url']
            else:
                print(f"ImgBB API Error: {result['error']['message']}")
                return None
        else:
            # Request failed
            print(f"ImgBB API Error: HTTP Status Code {response.status_code}")
            return None

if __name__ == '__main__':
    app.run(debug=True)
