#coding=utf-8
import os
from PIL import Image
import numpy as np



from keras.models import Sequential
from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.applications.inception_v3 import preprocess_input
from keras.layers.core import Dropout
import numpy as np
from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import optimizers
import keras
import time
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from picturesize import picturesize
from keras.models import load_model

'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')
'''
print ("ss")
model = load_model('vggtv4.h5')
for i, layer in enumerate(model.layers):
   print(i, layer.name)



'''
picturesize=picturesize()
model = Sequential()
model.add(Convolution2D(8, 5, 5, border_mode='valid',input_shape=(1,picturesize[0],picturesize[1])))
model.add(Activation('relu'))





model.add(Convolution2D(8, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16,3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, init='normal'))
model.add(Activation('relu'))

model.add(Dense(2, init='normal'))
model.add(Activation('softmax'))
'''


#base_model =InceptionV3(include_top=False,weights= 'imagenet') 

   
''' 
base_model =VGG16(include_top=False,weights= 'imagenet') 
vmodel=base_model.get_layer( index=17 ).output
vmodel=GlobalAveragePooling2D()(vmodel)
vmodel=Dense(256, activation='relu')(vmodel)                          
vmodel=Dropout(0.5)(vmodel)
vmodel=Dense(1, activation='sigmoid')(vmodel)                           
                           
                           
                           
model1 = Model(inputs=base_model.input, outputs=vmodel) 

model=Sequential()
model.add(model1)
for i, layer in enumerate(model.layers):
   print(i, layer.name)
   
k=model1.get_layer( index=18 ).input_shape
print(k)
'''                     
