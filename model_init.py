import numpy as np
import pandas as pd
import keras
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import tensorflow as tf

from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import clear_session
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def CNN_model(X,nClasses):
    clear_session()
    model_loss = 'categorical_crossentropy'
    kernel_window = 1
    pool_window = 1
    conv_model = Sequential()
    conv_model.add(Conv1D(32, kernel_size=kernel_window,activation='linear',
                          input_shape=(X.shape[1],X.shape[2]),padding='valid'))
    conv_model.add(LeakyReLU(alpha=0.1))
    conv_model.add(MaxPooling1D(pool_window,padding='valid'))
    conv_model.add(Dropout(.25))
    conv_model.add(Conv1D(64, kernel_size=kernel_window,activation='tanh',
                       input_shape=(X.shape[1],X.shape[2]),padding='valid'))
    conv_model.add(LeakyReLU(alpha=0.1))
    conv_model.add(MaxPooling1D(pool_window,padding='valid'))
    conv_model.add(Dropout(.4))
    conv_model.add(Flatten())
    conv_model.add(Dense(32, activation='linear'))
    conv_model.add(LeakyReLU(alpha=0.1))  
    conv_model.add(Dense(nClasses, activation='softmax'))
    conv_model.compile(loss=model_loss, optimizer='adam',metrics=['accuracy'])
    
    return conv_model

def LSTM_model(X,nClasses):
    clear_session()
    model_optimizer = 'rmsprop'
    metric = 'accuracy'
    model_loss = 'categorical_crossentropy'
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(180, 
                    input_shape=(None, X_train.shape[2]),
                    return_sequences=True,
                    return_state=False
                    ))

    model.add(keras.layers.LSTM(180, 
                        return_sequences=False,
                        return_state=False
                        ))
    model.add((keras.layers.Dense(nClasses,activation='softmax')))
    model.compile(loss=model_loss, optimizer=model_optimizer,metrics=[metric])
    return model

def MLP_model(X,nClasses):
    clear_session()
    model_loss = 'categorical_crossentropy'
    model = Sequential()
    model.add(Dense(64, activation='linear',input_shape=(X.shape[1],)))
    model.add(LeakyReLU(alpha=0.1))  
    model.add(Dense(32, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))  
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss=model_loss, optimizer='rmsprop',metrics=['accuracy'])
    
    return model

def LinearSVM_model(X,nClasses):
    model = SVC(kernel="linear", C=0.025)
    return model 

def LogReg_model(X,nClasses):
    model = LogisticRegression()
    return model