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
def KK(X,num_filter):
    #3*3+3*3
    filters=num_filter//2
    X1 = Conv_Leaky(X,num_filter)
    X1 = Conv_Leaky(X1,num_filter,stride_size=2)
    #1*1+3*3
    X2 = Conv_Leaky(X,num_filter,size=1)
    X2 = Conv_Leaky(X2,num_filter,stride_size=2)
    X3 = MaxPooling2D((2, 2))(X)
    out= concatenate([X1,X2,X3],axis=-1)
    out= Conv2D(num_filter,(1, 1), padding='same')(out)
    return out

def KK_T(X,num_filter):
    
    X1 = Conv_Leaky(X,num_filter)
    X1 = Conv_Leaky(X1,num_filter)
    X1 = UpSampling2D()(X1)
    X1 = Conv_Leaky(X1,num_filter)
    
    X2 = Conv_Leaky(X,num_filter,size=1)
    X2 = UpSampling2D()(X2)
    X2 = Conv_Leaky(X2,num_filter)
    
    X3 = UpSampling2D()(X)
    X3 = Conv2DTranspose(num_filter,(3, 3))(X3)
    
    out= concatenate([X1,X2,X3],axis=-1)
    out= Conv2D(num_filter,(1, 1), padding='same')(out)
    return out
 
def Res(X,num_filter): 
    X1 = Conv_Leaky(X,num_filter)
    X1 = Conv_Leaky(X1,num_filter)
    out = Add()([X1,X])
    return out

def Conv_Leaky(X,num_filter,padding='same',size=3,stride_size=1,alpha=0.2,momentum=0.8):
    X1 = Conv2D(num_filter,(size, size),padding=padding,strides=(stride_size,stride_size))(X)
    X1 = LeakyReLU(alpha=alpha)(X1)
    X1 = BatchNormalization(momentum=momentum)(X1)
    return X1