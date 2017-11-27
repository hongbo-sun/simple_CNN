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
reload(sys)
sys.setdefaultencoding('utf8')


picturesize=picturesize()
model = load_model('vggtv4.h5')
img = Image.open("1.jpg")
imgg = img.resize((picturesize[1],picturesize[0]))  # resize width*heght i.e.  column*rows





arr = np.asarray(imgg,dtype="float32")
data = np.empty((1,1,picturesize[0],picturesize[1]),dtype="float32")
data[0,:,:,:] = arr
data /= np.max(data)
data -= np.mean(data)
	









#define theano funtion to get output of  FC layer
get_feature = theano.function([model.layers[0].input],model.layers[12].output) 

#define theano funtion to get output of  first Conv layer 
get_featuremap = theano.function([model.layers[0].input],model.layers[6].output) 




# visualize feature  of  Fully Connected layer
#data[0:10] contains 10 images
feature = get_feature(data)  #visualize these images's FC-layer feature
plt.imshow(feature,cmap = cm.Greys_r)
print (feature)
plt.colorbar()  
plt.show()


#visualize feature map of Convolution Layer
num_fmap = 4	#number of feature map
for i in range(num_fmap):
    
    featuremap = get_featuremap(data)
    print (featuremap)
    plt.imshow(featuremap[0][i],cmap = cm.Greys_r) #visualize the first image's 4 feature map
    plt.colorbar()  
    plt.show()
