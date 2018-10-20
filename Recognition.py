''' importing libraries '''

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation,Input,concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from PIL import Image

np.set_printoptions(threshold=np.nan)

''' Facenet '''

FRmodel = faceRecoModel(input_shape=(3,96,96))

# Getting the parameters of the model

print("Total Params:",FRmodel.count_params())

#Triplet Loss

def triplet_loss(y_true,y_pred,alpha=0.2):

    anchor,positive,negetive = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negetive)),axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss,0))
    return loss

## Compiling the model and loading weights

FRmodel.compile(optimizer='adam',loss=triplet_loss,metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)


## Database

user_name = "Abhishek Jha"
database = {}

database["Abhishek Jha"] = img_to_encoding("Images/Abhi.jpg",FRmodel)


def get_identity(image_path,database,model):
 
    identity = "Unknown"
    id_temp = None
    min_dist = 999
    img_encoding = img_to_encoding(image_path,model)
    for name in database.keys():

        dist = np.linalg.norm(img_encoding-database[name])
        if dist<min_dist:
            min_dist = dist
            id_temp = name

    if min_dist < 0.7:
        identity = id_temp

    return identity




def capture_image():

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0
    flag=1
   
    while True:
        ret, frame = cam.read()
        cv2.imshow("Press SPACE to take a picture and ESC to close", frame)
        if not ret:
          break
        k = cv2.waitKey(1)

        if k%256 == 27:
          print("Camera Closed")
          break
        elif k%256 == 32:
          img_name = "webcam_{}.png".format(img_counter)
          cv2.imwrite(img_name, frame)
          res = cv2.imread(img_name)
          img = Image.fromarray(res, 'RGB')
          img.save('Images/webcam.jpg')    
          print("{} has been added".format(img_name))
          img_counter += 1
          flag=0
    
    cam.release()

    cv2.destroyAllWindows()
    

## crop and resize

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('Images/webcam.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
       cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
       roi_gray = gray[y:y+h, x:x+w]
       roi_color = img[y:y+h, x:x+w]

    crop_img = img[y:y+h, x:x+w]
    res = cv2.resize(crop_img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(res, 'RGB')
    img.save('Images/webcam.jpg')
    img_path = 'Images/webcam.jpg'

    return img_path


    
