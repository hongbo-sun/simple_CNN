#coding:utf-8
from __future__ import print_function
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
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
	picturefolder="T1"
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




def svm(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    print (pred_testlabel)
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
    
    train_number=180
    (traindata,testdata) = (data[0:train_number],data[train_number:])
    (trainlabel,testlabel) = (label[0:train_number],label[train_number:])
    #use origin_model to predict testdata
    origin_model = load_model("4.h5")
    #print(origin_model.layers)
    pred_testlabel = origin_model.predict_classes(testdata,batch_size=1, verbose=1)
    num = len(testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print(" Origin_model Accuracy:",accuracy)
    
    #define theano funtion to get output of specified layer of the model
    get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[10].output,allow_input_downcast=False)
    
    feature = get_feature(data)
    
    weidu=feature.ndim     #如果维度超过2，要把数据化成2维，因为scaler.fit_transform只能处理二维数组，feature_vector：第一维表示第几个数据，第二维就是相应的特征
    if weidu>2:
        shape=np.shape(feature)
        feature_vector = np.empty((shape[0],shape[1]*shape[2]*shape[3]),dtype="float32")
        for i in range(shape[0]):
            feature_vector [i] = feature[i].flatten()     
    else:
        feature_vector=feature
            
        
    #train svm using obtained feature
    scaler = MinMaxScaler()
    feature_vector = scaler.fit_transform(feature_vector)#使用这种标准化方法的原因是，有时数据集的标准差非常非常小，有时数据中有很多很多零（稀疏数据）需要保存住０元素。
    svm(feature_vector[0:train_number],label[0:train_number],feature_vector[train_number:],label[train_number:])
