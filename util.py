import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

def train_validation_split(X,y,classes,dataframe_tot,model):
    
    if model in ['CNN','RNN','MLP']:
        y = to_categorical(y)
        train_ix = []
        test_ix = []
        train_size = 0.75
        for app in classes:
            ix = list(dataframe_tot[dataframe_tot.app == app].index)
            train_ind = int(len(ix)*train_size)
            train_ix.append(ix[:train_ind])
            test_ix.append(ix[train_ind:])

        X_train = X[train_ix[0]]
        X_test = X[test_ix[0]]
        y_train = y[train_ix[0]]
        y_test = y[test_ix[0]]
        for ind in range(len(classes)-1):
            X_train = np.vstack([X_train,X[train_ix[ind+1]]])
            X_test = np.vstack([X_test,X[test_ix[ind+1]]])
            y_train = np.vstack([y_train,y[train_ix[ind+1]]])
            y_test = np.vstack([y_test,y[test_ix[ind+1]]])
    else:
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train,X_test,y_train,y_test

def get_ood_diff(scores,scores_test):
 
    cdf_diff =  []
    cdf_min = []
    b = 50
    hist_unlabel = np.zeros([3,b])
    hist_label = np.zeros([3,b])
    for ind in [0,1,2]:
        plt.clf()
        hist_label[ind,:] = [h.get_height() for h in sns.distplot(scores_test[ind,:],
                     bins=b,
                    hist_kws=dict(cumulative=True),
                     kde_kws=dict(cumulative=True)).patches]
        plt.clf()
        hist_unlabel[ind,:] = [h2.get_height() for h2 in sns.distplot(scores[ind,:],
                     bins=b,
                    hist_kws=dict(cumulative=True),
                     kde_kws=dict(cumulative=True)).patches]
        plt.close()
    m = np.mean(abs(hist_label - hist_unlabel),axis=1)
    cdf_diff.append(np.mean(m)/4)
    return cdf_diff

def bar_plot_confidence(array_tot,cdf_diff,cell):
    fig = plt.figure(figsize=[8,6])
    ax = fig.add_subplot(111)
    app_types = ['video-streaming','audio-streaming', 'video-calls']

    f= {'campnou':[.3,.3,1.5],
        'paolo': [.2,.2,.8] }
    
    ind = np.arange(max(array_tot.shape))*2
    alpha_val = 0.7
    width = 2
    fs = 15
    
    if array_tot.shape[0]>3:
        array_tot = array_tot*f[cell]
        array_tot = array_tot.transpose()
        
    composition = array_tot/sum(array_tot)*100
    composition[0,:] = composition[0,:] - cdf_diff*100*.7
    composition[1,:] = composition[1,:] - cdf_diff*100*.3
    
    plt.bar(ind, array_tot[0,:], align="center", width=width,edgecolor='w')
    plt.bar(ind, array_tot[1,:], align="center", width=width, bottom = array_tot[0,:],edgecolor='w')
    plt.bar(ind, array_tot[2,:], align="center", width=width, bottom = array_tot[0,:]+array_tot[1,:],edgecolor='w')
    
    plt.bar(ind,cdf_diff*np.sum(array_tot,axis=0), align="center", width=width, color='grey',
            bottom = array_tot[0,:]+array_tot[1,:]+array_tot[2,:],edgecolor='w')
    
    plt.tight_layout()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fs)
    plt.ylabel('Session Count',fontsize = fs)
    plt.xlabel('Time [h]',fontsize = fs)
    plt.legend(['video-streaming','audio-streaming', 'video-calls','OOD'],
               ncol=2,fontsize=fs,loc = 'upper left')
    
    fs = 12
    for n in range(len(ind)):
        if array_tot[0,n] > 20:
            plt.annotate('%.0f%%'%composition[0,n], (ind[n],0), (ind[n], array_tot[0,n]/2-fs/2), 
                         ha='center',   size=fs, color='w')
        if array_tot[1,n] > 15:
            plt.annotate('%.0f%%'%composition[1,n], (ind[n],0), (ind[n], array_tot[0,n]+array_tot[1,n]/2), 
                         ha='center',   size=fs, color='w')
        if array_tot[2,n] > 15:
            plt.annotate('%.0f%%'%composition[2,n], (ind[n],0), (ind[n], array_tot[0,n]+array_tot[1,n]+array_tot[2,n]/3-fs/4), 
                         ha='center',   size=fs, color='w')
        else:
            y_coord = array_tot[0,n]+array_tot[1,n]+array_tot[2,n] + 5
        if cdf_diff[n]*sum(array_tot[:,n]) > 15:
            val = int(cdf_diff[n]*100)
            plt.annotate('%.0f%%'%val, (ind[n],0), (ind[n], array_tot[0,n]+array_tot[1,n]+array_tot[2,n]
                                                                +val-fs/4), ha='center',   size=fs, color='w')
        else:
            val = int(cdf_diff[n]*100)
            y_coord = array_tot[0,n]+array_tot[1,n]+array_tot[2,n]+val + 10
            plt.plot([ind[n], ind[n]], [array_tot[0,n]+array_tot[1,n]+array_tot[2,n]+val/4, y_coord], 'k--',linewidth=1)
            plt.annotate('%.0f%%'%val, (ind[n],0), (ind[n], y_coord), ha='center',   size=fs, color='k')
    return composition

