# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:53:20 2019

@author: hongb
"""

import os
from keras.utils import plot_model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense,Flatten,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from keras.layers.core import Dropout



from keras import backend as K
import numpy as np



os.environ['CUDA_VISIBLE_DEVICES']='0'



def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a image.

    # Arguments
        x: input Numpy tensor of a image, 3D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """ 
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        #x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= np.mean(x[ 0, :, :])
        x[ 1, :, :] -= np.mean(x[ 1, :, :])
        x[ 2, :, :] -= np.mean(x[ 2, :, :])
    else:
        # 'RGB'->'BGR'
        #x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= np.mean(x[ :, :, 0])
        x[:, :, 1] -= np.mean(x[ :, :, 1])
        x[:, :, 2] -= np.mean(x[ :, :, 2])
       
    return x











class PowerTransferMode:
    #数据准备
    def DataGen(self, dir_path, img_row, img_col, batch_size, is_train):
        if is_train:
            datagen = ImageDataGenerator(rescale=1./255,
                                         #preprocessing_function=preprocess_input
                                         )
        else:
            datagen = ImageDataGenerator(rescale=1./255,
                                         #preprocessing_function=preprocess_input
                                         )
 
        generator = datagen.flow_from_directory(
            dir_path, target_size=(img_row, img_col),
            batch_size=batch_size,
            
            #class_mode='binary',
            
            shuffle=is_train)
 
        return generator

 
 
    #VGG模型
    def VGG19_model(self, lr=0.01, decay=1e-6, momentum=0.9, nb_classes=200, img_rows=256, img_cols=256, RGB=True, is_plot_model=False):
        color = 3 if RGB else 1
        base_model = VGG19(weights='imagenet', include_top=False, pooling=None, input_shape=(img_rows, img_cols, color),
                              classes=nb_classes)
 
        #冻结base_model所有层，这样就可以正确获得bottleneck特征
        for layer in base_model.layers:
            layer.trainable = False
 
        x = base_model.output
        #添加自己的全链接分类层
        x = GlobalAveragePooling2D()(x)
        #x = Dense(, activation='relu')(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
 
        #训练模型
        model = Model(inputs=base_model.input, outputs=predictions)
        sgd = SGD(lr=lr, momentum=momentum)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 
        # 绘图
        if is_plot_model:
            plot_model(model, to_file='vgg19_model.png',show_shapes=True)
 
        return model
 

 
    #训练模型
    def train_model(self, model, epochs, train_generator, steps_per_epoch, validation_generator, validation_steps, model_url, is_load_model=False):
        # 载入模型
        if is_load_model and os.path.exists(model_url):
            model = load_model(model_url)
            for layer in model.layers:
                layer.trainable = True
				
            #for layer in model.layers[16:]:
                #layer.trainable = True
            sgd = SGD(lr=1e-4,  momentum=0.9)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            
            
            
            
 
        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps)
        # 模型保存
        model.save(model_url,overwrite=True)
        return history_ft
 
    # 画图
    def plot_training(self, history):
      acc = history.history['acc']
      val_acc = history.history['val_acc']
      loss = history.history['loss']
      val_loss = history.history['val_loss']
      epochs = range(len(acc))
      plt.plot(epochs, acc, 'b-')
      plt.plot(epochs, val_acc, 'r')
      plt.title('Training and validation accuracy')
      plt.figure()
      plt.plot(epochs, loss, 'b-')
      plt.plot(epochs, val_loss, 'r-')
      plt.title('Training and validation loss')
      plt.show()
 
 
if __name__ == '__main__':

 
    transfer = PowerTransferMode()
    train_num=32000
    val_num=8000
    #epo=10
    train_dir='make_train'
    val_dir='make_val'
 
    #得到数据
    train_generator = transfer.DataGen(train_dir, 256, 256, 64, True)
    validation_generator = transfer.DataGen(val_dir, 256, 256, 64, False)
 
    #VGG19
    #model = transfer.VGG19_model(nb_classes=200, img_rows=256, img_cols=256, is_plot_model=False)
    base_model=VGG19(include_top=False,weights= 'imagenet',input_shape=(256,256,3)) 
    vmodel=base_model.get_layer(index=20).output
    vmodel=GlobalAveragePooling2D()(vmodel)
    vmodel=Dropout(0.5)(vmodel)
    vmo=Dense(200, activation='softmax')(vmodel)                                      
    model=Model(inputs=base_model.input, outputs=vmo) 
    model.summary()
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True
    sgd = SGD(lr=1e-2,  momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	
    
    #训练新加层
    start1=time.time()
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    history_ft = transfer.train_model(model, 15, train_generator, train_num/64, validation_generator, val_num/64, 'vgg19_model_weights.h5', is_load_model=False)
    end1=time.time()
    print('调新加全连接层所用时间 cost %4.4fs' % ((end1-start1)))
    
    
    #网络后面部分的微调(最后一个block和全连接层)
    start2=time.time()
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)
    history_ft1 = transfer.train_model(model, 15, train_generator, train_num/64, validation_generator, val_num/64, 'vgg19_model_weights.h5', is_load_model=True)
    end2=time.time()
    print('微调所用时间 cost %4.4fs' % ((end2-start2)))
 

    #InceptionV3
    #model = transfer.InceptionV3_model(nb_classes=2, img_rows=image_size, img_cols=image_size, is_plot_model=True)
    #history_ft = transfer.train_model(model, 10, train_generator, 600, validation_generator, 60, 'inception_v3_model_weights.h5', is_load_model=False)
 
    # 训练的acc_loss图
    transfer.plot_training(history_ft)
    transfer.plot_training(history_ft1)
