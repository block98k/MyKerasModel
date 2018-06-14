from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Lambda
from keras import backend as K
from keras import metrics
from keras import optimizers
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D,Add
from keras.layers import Conv2DTranspose,Reshape,concatenate,Reshape
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras.optimizers import Adam
from My_Block import *
def create_encoder(input_size,latten_dim):
    net_input = Input(shape=(input_size,input_size,3))
    X = Conv_Leaky(net_input,16)
    X = Conv_Leaky(X,16)
    X = MaxPooling2D((2, 2))(X)
    filters=32
    for i in range(6):
        X=kk_block(X,filters)
        filters*=2
    X = Conv_Leaky(X,latent_dim)
    return Model(net_input,X)

def create_decoder(latent_dim,middle_dim,activations='sigmoid'):
    Z_input = Input(shape=(1,1,latent_dim))
    X = Conv2DTranspose(middle_dim,(3, 3),padding='valid',activation='relu')(Z_input)
    X = BatchNormalization(momentum=0.8)(X)
    filters=1024
    for i in range(6):
        X=kk_block_T(X,filters)
        filters//=2
    X = Conv_Leaky(X,16)
    X = UpSampling2D()(X)
    X = Conv_Leaky(X,16)
    X = Conv_Leaky(X,16)
    X = Conv2D(3,(3, 3))(X)
    X = BatchNormalization()(X)
    out = Activation(activations)(X)
    return Model(Z_input,out)

def create_discriminator(input_size,activations='sigmoid'):
    net_input = Input(shape=(input_size,input_size,3))
    X = Conv_Leaky(net_input,32)
    X = Conv_Leaky(X,32)
    X = MaxPooling2D((2, 2))(X)
    filters=64
    for i in range(5):
        X=kk_block(X,filters)
        filters*=2
    X = Conv_Leaky(X,128)
    X = Flatten()(X)
    validity=Dense(1, activation=activations)(X)
    return Model(net_input,validity)

def create_mnist():
    inputs = Input(shape=(28,28,1))
    x = Conv_Leaky(inputs,32,padding='valid')
    x = Conv_Leaky(x,64,padding='valid')
    x = Res(x,64)
    x = Res(x,64)
    x = MaxPooling2D()(x)
    x = Conv_Leaky(x,128,padding='valid')
    x = Conv_Leaky(x,128,padding='valid')
    x = Conv_Leaky(x,256,padding='valid')
    x = Conv_Leaky(x,256,padding='valid')
    x = Flatten()(x)
    out = Dense(10,activation='softmax')(x)
    return  Model(inputs,out)