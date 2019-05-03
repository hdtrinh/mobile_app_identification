import pandas as pd
import numpy as np
import os, time, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import datetime as dt
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

import util

sns.set_style("darkgrid")

import warnings
warnings.filterwarnings('ignore')


def load_input(filename):
    max_len  = 80
    df_unsupervised = pickle.load(open(filename,'rb'))
    X = df_unsupervised.iloc[:,0:max_len]
    X = StandardScaler().fit_transform(X.transpose()).transpose()
    X = X.reshape(X.shape[0],X.shape[1],1)
    return X

def get_scores(X):
    d = model.predict(X)
    scores = np.array([d[:,0] + d[:,2],d[:,1] + d[:,4],d[:,3] + d[:,5]])
    return scores

def get_composition(filename):
    X = load_input(filename)
    prediction = model.predict(X)
    d = Counter(np.argmax(np.round(prediction),axis=1))
    composition = np.array([d[0] + d[2],d[1] + d[4],d[3] + d[5]])
    return sorted(composition,reverse=True)

def ood_diff(f):
    X = load_input(f)
    scores = get_scores(X)
    return util.get_ood_diff(scores,scores_test)

from keras.models import load_model
classes = ['googlemusic', 'skype', 'spotify', 'vimeo', 'whatsappvideo','youtube']
model = load_model('cnn_model.h5')
max_len  = 80
X_test, y_test, d_name= pickle.load(open('cnn_test_sets.pkl','rb'))
scores_test = get_scores(X_test[:,0:max_len])


cell_name = ['campnou','paolo']
composition_dict = dict()
OOD_dict = dict()
for cell in cell_name:
    print('Traffic composition for {} cell'.format(cell))
    OOD_list = []
    folder = '../use_case_files/'
    filenames = [f for f in os.listdir(folder) if 'df_' + cell in f]
    filepaths = [os.path.join(folder,name) for name in sorted(filenames)]
    for num,f in enumerate(filepaths):
        print('Slot {}/12'.format(num+1))
        if num == 0:
            array_tot = get_composition(f)
        else:
            array_temp = get_composition(f)
            array_tot = np.vstack([array_tot, array_temp])
        OOD_list.append(np.mean(ood_diff(f)))
    composition_dict[cell] = array_tot
    OOD_dict[cell] = np.array(OOD_list)
for cell in cell_name:
    util.bar_plot_confidence(composition_dict[cell],OOD_dict[cell],cell)
    plt.ylim([0,450])
    plt.title(cell,size=15)
    plt.savefig(cell + '_composition.pdf',bbox_inches='tight')