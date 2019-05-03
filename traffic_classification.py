import numpy as np
import pandas as pd
import keras
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import tensorflow as tf
import model_init
import util

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.backend import clear_session
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

app_type = {'vimeo':'video_streaming',
            'spotify':'audio',
            'youtube':'video_streaming',
            'skype':'video_call',
            'whatsappvideo':'video_call',
           'googlemusic':'audio'}

app_type_class = {'video_streaming':0,
                  'video_call':1,
                  'audio':2}

# load data
def get_X (dataframe_tot,max_len,model):
    X = dataframe_tot.iloc[:,0:max_len]
    X = StandardScaler().fit_transform(X.transpose()).transpose()
    if model == 'CNN' or model == 'RNN':
        X = X.reshape(X.shape[0],X.shape[1],1)
    return X

def get_train_validation_sets():
    return util.train_validation_split(X,y,classes,df_sessions,model)

# get model
def get_model(model):
    if model == 'CNN':    
        return model_init.CNN_model(X,nClasses)
    if model == 'RNN':
         return model_init.CNN_model(X,nClasses)
    if model == 'MLP':             
        return model_init.MLP_model(X,nClasses)
    if model == 'LinearSVM':             
        return model_init.LinearSVM_model(X,nClasses)
    if model == 'LogReg':             
        return model_init.LogReg_model(X,nClasses)

def train_model(chosen_model,model, X_train, y_train,X_test, y_test, 
                batch = 64,n_it=20):
    if model in ['CNN','MLP','RNN']:
        trained_model = chosen_model.fit(X_train, y_train, 
                                   batch_size=batch,
                                   epochs=n_it,
                                   verbose=1,
                                   validation_data=(X_test, y_test))
    else:
        trained_model = chosen_model.fit(X_train, y_train)
    return trained_model

def get_service_accuracy(model,trained_model,chosen_model,X_test,y_test,y_pred):
    if model in ['CNN','MLP','RNN']:
        y_pred = np.argmax(np.round(chosen_model.predict(X_test)),axis=1)
        y_test = np.argmax(y_test,axis=1)
    else:
        y_pred = chosen_model.predict(X_test)

    y_pred_app = lab_enc.inverse_transform(y_pred)
    y_test_app = lab_enc.inverse_transform(y_test)
    y_pred = []
    y_test = []
    for y_t,y_p in zip(y_test_app,y_pred_app):
        y_pred.append(app_type_class[app_type[y_p]])
        y_test.append(app_type_class[app_type[y_t]])
    cm = confusion_matrix(y_test, y_pred)
    accuracy = cm.diagonal().sum()/cm.sum()
    precision,recall,fscore,_ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return accuracy,precision,recall,fscore

def get_app_accuracy(model,trained_model,chosen_model,X_test,y_test,y_pred):
    if model in ['CNN','MLP','RNN']:
        score = np.mean(trained_model.history['val_acc'])
        y_pred = np.argmax(np.round(chosen_model.predict(X_test)),axis=1)
        y_test = np.argmax(y_test,axis=1)
    else:
        score = chosen_model.score(X_test, y_test)
        y_pred = chosen_model.predict(X_test)
        
    precision,recall,fscore,_ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return score,precision,recall,fscore

filename = 'sessions_df.pkl'
df_sessions = pickle.load(open(filename,'rb'))
num_of_sessions = len(df_sessions)
y = df_sessions['app']
classes = np.unique(y)
nClasses = len(classes)

# Hot-encoding output y
lab_enc = LabelEncoder().fit(y)
y = lab_enc.transform(y)

# For loops
sess_lenghts = [5,10,20,30,40,50,60]
accuracy_list = []
accuracy_dict = {}

problem = 'app_classification'  # app_classification or service_classification
date_string = dt.datetime.now().strftime('%Y%m%d%H%M%S')
classifiers = [
                'CNN',
               'RNN',
               'MLP',
                'LinearSVM',
                'LogReg'
              ]

log = ''

for num,model in enumerate(classifiers):
    print('Training {} model for {}'.format(model,problem))
    accuracy_list = []
    log = log + 'Training {} model for {}\n'.format(model,problem)
    for max_len in sess_lenghts:
        if model in ['CNN','MLP','RNN']:
            print('Training {} model for {}. Session length: {}'.format(model,problem,max_len))
        X = get_X(df_sessions,max_len,model)
        X_train,X_test,y_train,y_test = get_train_validation_sets()
        chosen_model = get_model(model)
    
        trained_model = train_model(chosen_model,model,
                                    X_train, y_train,
                                    X_test, y_test)

        if problem == 'app_classification':
            accuracy,precision,recall,fscore = get_app_accuracy(model,trained_model,chosen_model,
                                                                X_test,y_test,y_train)
        else:
            accuracy,precision,recall,fscore = get_service_accuracy(model,trained_model,chosen_model,
                                                                X_test,y_test,y_train)

        accuracy_list.append(accuracy)
        print('Session length: {} - Accuracy: {:.2f}\n'.format(max_len,accuracy*100))
        log = log + 'Session length: {} - Accuracy: {:.2f}\n'.format(max_len,accuracy*100)
    accuracy_dict[model] = accuracy_list
    print(log)
    marker_size = 8
    filled_markers = ( 'v', '^', '<','o','p') 
    sns.set_style('whitegrid')

    #plt.plot(sess_lenghts,[a*100 for a in accuracy_list],
     #        linestyle = '-',
     #        marker = filled_markers[num],
     #        markersize = marker_size,
     #        markerfacecolor='w',
     #        markeredgewidth=1.5)

    #plt.xlabel('Session length',fontsize = 12)
    #plt.ylabel('Accuracy %',fontsize = 12)
    #plt.title('{}'.format(problem))
#plt.legend(classifiers)
print('Saving plot in current folder')
#plt.savefig('accuracy_vs_session_length.pdf',bbox_inches='tight')