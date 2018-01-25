#coding=utf-8
from __future__ import print_function
import theano
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.models import load_model
from PIL import Image
import numpy as np
from picturesize import picturesize
#load the saved model



import sys
sys.setrecursionlimit(1000000000)


picturesize=picturesize()
model = load_model('vggms_without_concat.h5')
img = Image.open("2.jpg")
imgg = img.resize((picturesize[1],picturesize[0]))  # resize width*heght i.e.  column*rows

'''
img1 = Image.open("1.bmp")
plt.imshow(img1) 
plt.show()
'''




arr = np.asarray(imgg,dtype="float32")
data = np.empty((1,3,picturesize[0],picturesize[1]),dtype="float32")
data[0,0,:,:] = arr[:,:,0]-np.mean(arr[:,:,0])
data[0,1,:,:] = arr[:,:,1]-np.mean(arr[:,:,1])
data[0,2,:,:] = arr[:,:,2]-np.mean(arr[:,:,2])







#define theano funtion to get output of  FC layer
#get_feature = theano.function([model.layers[0].input],model.layers[10].output) 

#define theano funtion to get output of  first Conv layer 
get_featuremap = theano.function([model.layers[0].input],model.layers[4].output) 




# visualize feature  of  Fully Connected layer
#feature = get_feature(data)  #visualize these images's FC-layer feature
#plt.figure("final_probability") 
#plt.imshow(feature,cmap = cm.Greys_r)
#print (feature)
#plt.colorbar() 
#plt.show()



featuremap = get_featuremap(data)
print(featuremap.shape)
shape=featuremap.shape
#visualize feature map of Convolution Layer
num_fmap =shape[1] 	#number of feature map
for i in range(num_fmap):
    

    #print (featuremap)
    name=str(i)+"-th_feature_map"
    plt.figure(name)
    plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
    plt.colorbar()  


    plt.show()
