#coding:utf-8

from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range
import random
from keras.callbacks import EarlyStopping
import numpy as np
from keras import metrics
from keras.layers import Conv2D, MaxPooling2D,Input,Dense,Activation,Flatten
from keras.models import load_model,Model

import os
from PIL import Image
import numpy as np

import os
from PIL import Image
import numpy as np

from keras.callbacks import ModelCheckpoint

import sys
import matplotlib.pyplot as plt
from sklearn import preprocessing

from picturesize import picturesize
#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，如果是将彩色图作为输入,则将1替换为3,图像大小28*28

class precisionsHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.precisions = []
        self.recalls = []
        self.accs = []
    def on_epoch_end(self, batch, logs={}):
        self.precisions.append(logs.get('val_precision'))
        self.recalls.append(logs.get('val_recall'))
        self.accs.append(logs.get('val_acc'))


  
  
#导入各种用到的模块组件
reload(sys)
sys.setdefaultencoding('utf8')
sys.setrecursionlimit(1000000000)
picturesize=picturesize()
def load_data():
	picture_num=202
	picturefolder="C:\Users\lenovo\Desktop/T1"
	data = np.empty((picture_num,1,picturesize[0],picturesize[1]),dtype="float32")
	label = np.empty((picture_num,),dtype="uint8")
	imgs = os.listdir(picturefolder)
	num = len(imgs)
	#print(imgs)
	#os.system("pause")
	for i in range(num):
		img = Image.open(picturefolder+"/"+imgs[i])
		imgg = img.resize((picturesize[1],picturesize[0]))  # resize width*heght i.e.  column*rows
		#imgg.show()
		arr = np.asarray(imgg,dtype="float32")
		data[i,:,:,:] = arr
		label[i] = int(imgs[i].split(' ',1)[0])
	#归一化和零均值化
	data /= np.max(data)
	data -= np.mean(data)
	return data,label





#加载数据
data, label = load_data()


#label为0~9共10个类别，keras要求形式为binary class matrices,转化一下，直接调用keras提供的这个函数
nb_class = 2
label =np_utils.to_categorical(label, nb_class)  #(label,nb_class)

#序贯式模型
def create_model():
    # type: () -> object
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

	model.add(Dense(nb_class, init='normal'))
	model.add(Activation('softmax'))
	return model



#函数式模型
def create_model1():
   input_image=Input(shape=(1,picturesize[0],picturesize[1])) 
   x0=Conv2D(8,(5,5),padding='valid',activation='relu')(input_image)
   x1=Conv2D(8,(3,3),padding='valid',activation='relu')(x0)
   x2=MaxPooling2D((2,2))(x1)
   x3=Conv2D(16,(3,3),padding='valid',activation='relu')(x2)


   x4=MaxPooling2D((2,2))(x3)

   x5=Flatten()(x0)
   x6=Flatten()(x4)

   x7 = keras.layers.concatenate([x5, x6], axis=-1)
   #x7= preprocessing.normalize(x7, norm='l2')
   #x8=Flatten()(x7)
   x9=Dense(128,activation='relu')(x7)
   x10=Dense(2,activation='softmax')(x9)
   model =Model(inputs=input_image,outputs=x10)
   return model

     
#############
#开始训练模型
##############
model = create_model1()
sgd = SGD(lr=0.01,  momentum=0.9) #decay=1e-6,nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
train_num=150
(X_train,X_val) = (data[0:train_num],data[train_num:])
(Y_train,Y_val) = (label[0:train_num],label[train_num:])

#使用early stopping返回最佳epoch对应的model
#early_stopping = EarlyStopping(monitor='val_loss', patience=1)
history = precisionsHistory()
checkpointer = ModelCheckpoint(monitor='val_acc',filepath='vggtv4.h5', verbose=1, save_best_only=True)
jilu=model.fit(X_train, Y_train, batch_size=32,validation_data=(X_val, Y_val),nb_epoch=10,verbose=1,callbacks=[checkpointer,history])
 
#保存最后一次训练后的模型
model.save('4.h5')  

 
 
 
 
 
#画loss和val_loss曲线
acc = jilu.history['acc']
val_acc = jilu.history['val_acc']
epochs = range(len(acc))
#plt.axis([0,5,0,1])
plt.grid(True)
plt.plot(epochs, acc, 'r',label='acc')
plt.plot(epochs, val_acc, 'g',label='val_acc')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.ylim(0,1)
plt.title('Training and validation accuracy')
plt.savefig("train.png")
plt.show()




