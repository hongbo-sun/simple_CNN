#coding:utf-8
'''
Author:wepon
Code:https://github.com/wepe

File: cnn-svm.py
'''
from __future__ import print_function
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from data import load_data
import random
from keras.models import load_model
from picturesize import picturesize
import numpy as np
import os
from PIL import Image

import sys
reload(sys)
sys.setdefaultencoding('utf8')

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




def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)
    
    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

def rf(traindata,trainlabel,testdata,testlabel):
    print("Start training Random Forest...")
    rfClf = RandomForestClassifier(n_estimators=400,criterion='gini')
    rfClf.fit(traindata,trainlabel)
    
    pred_testlabel = rfClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-rf Accuracy:",accuracy)

if __name__ == "__main__":
    #load data
    data, label = load_data()
    #shuffle the data
    index = [i for i in range(len(data))]
    random.shuffle(index)
    data = data[index]
    label = label[index]
    
    train_number=100
    (traindata,testdata) = (data[0:train_number],data[train_number:])
    (trainlabel,testlabel) = (label[0:train_number],label[train_number:])
    #use origin_model to predict testdata
    origin_model = load_model("4.h5")
    #print(origin_model.layers)
    pred_testlabel = origin_model.predict_classes(testdata,batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(" Origin_model Accuracy:",accuracy)
    #define theano funtion to get output of FC layer
    get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[10].output,allow_input_downcast=False)
    feature = get_feature(data)
    #train svm using FC-layer feature
    scaler = MinMaxScaler()
    feature = scaler.fit_transform(feature)
    rf(feature[0:train_number],label[0:train_number],feature[train_number:],label[train_number:])
