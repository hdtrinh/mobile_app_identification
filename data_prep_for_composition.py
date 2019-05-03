import pandas as pd
import numpy as np
import os, time, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from IPython.display import clear_output
import datetime as dt


# params
cell = 'campnou'            # cell name
file_folder  = '/results/'  # path to sniffer output
day = '20190127'            # day to be analyzed format YYYYMMDD
# ---------


dci_columns = ['subframe_n','subframe_ind','rnti','direction',
               'mcs','rbs','tbs','tbs_cw0','tbs_cw1',
               'dci_type','new_data_cw0','new_data_cw1',
               'harq_id','ncce','agg_level','cfi','corr_check']

day_slots = ['00','02','04','06','08','10','12',
             '14','16','18','20','22','24']

ts = [day + ds for ds in day_slots]
te = ts[1:]
ts = ts[:-1]

for time_start,time_end in zip(ts,te):
    folder = file_folder
    print(time_start,time_end)
    file_names = os.listdir(folder)
    file_list = sorted([os.path.join(folder, filename) for filename in file_names 
                        if ((filename >= time_start) and (filename < time_end))])
    print('files',len(file_list))
    if len(file_list)<1:
        continue

    df = pd.concat((pd.read_csv(f,header=None,sep='\t',names=dci_columns) for f in file_list)) 
    rnti_series = df.groupby('rnti').tbs.sum()
    top_rnti = [r for r in rnti_series.nlargest(n=100).index if r > 10]

    rnti = top_rnti
    rnti_dict = dict()
    for r in rnti:
        rnti_dict[r] = pd.DataFrame()
    for num,f in enumerate(file_list):
        if np.mod(num,720)==0:
            print(num,'/',len(file_list))
        temp_df = pd.read_csv(f,header=None,sep='\t',names=dci_columns)
        for r in rnti:
            temp_rnti_df = temp_df[temp_df.rnti==r]
            if len(temp_rnti_df)>0:
                temp_rnti_df['date'] = os.path.basename(f)[0:14]
                rnti_dict[r] = pd.concat([rnti_dict[r],temp_rnti_df])
    
    print('create rnti dict: done')

    for r in rnti:
        rnti_dict[r].index = [pd.to_datetime(d) for d in rnti_dict[r].date]

    filename = 'use_case_files/rnti_dict_'+cell+'_'+time_start+'_'+time_end+'.pkl'
    
    
    folder = 'use_case_files/unsup_'+cell+'_' + time_start+ '_'+time_end+'/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    def trace_split(df,sess_app):
        start_ix = []
        end_ix = []
        prev_tbs = 0
        Kb = 1000
        data_thr = 10
        duration_thr_max = 60
        duration_thr_min = 0
        zero_thr = 10 #seconds
        zero_count = 0
        silence = True
        # FOR - SPLITTER
        for num,row in enumerate(df.tbs):
            diff = np.abs(prev_tbs-row)
            prev_tbs = row
            if silence and diff < data_thr:
                continue

            if silence and diff >= data_thr:
                silence = False
                if num>0:
                    start_ix.append(num-1)
                else:
                    start_ix.append(num)
        #IF NOT SILENT
            if diff < data_thr:
                if zero_count == 0:
                    end_num = num
                zero_count = zero_count + 1
            else:
                zero_count = 0

            if zero_count > zero_thr:
                zero_count = 0
                silence = True
                if end_num>0:
                    end_ix.append(end_num)

        end_ix.append(num)

        duration_min = 0
        duration_max = 999
        count = 0
        start_cut = dt.datetime.strptime('1991-10-03 05:40:00','%Y-%m-%d %H:%M:%S')
        end_cut = dt.datetime.strptime('2100-10-20 03:50:00','%Y-%m-%d %H:%M:%S')

        sess_list = []

        for s,e in zip(start_ix,end_ix):
            duration = e - s
            if duration >= duration_min and duration <= duration_max:
                sess = df.iloc[s:e]
                if len(sess)<1:
                    continue
                start_time = sess.index[0]
                end_time = sess.index[-1]
                if start_time > start_cut and end_time < end_cut:
                    count = count + 1
                    fn = 'sess_' + start_time.strftime('%Y-%m-%d_%H:%M:%S') + '_' + end_time.strftime('%Y-%m-%d_%H:%M:%S')
                    sess.to_pickle(folder+fn+'_'+sess_app+'.pkl')
                    sess_list.append(sess)
        return count

    for r in rnti:
        tempdf = rnti_dict[r]
        tempdf = tempdf[tempdf.tbs<100000].tbs
        tempdf = tempdf.groupby(tempdf.index).sum()
        new_idx = pd.date_range(start=tempdf.index[0], end=tempdf.index[-1],freq='S')
        df = pd.DataFrame(index=new_idx).join(pd.DataFrame(tempdf)).fillna(0) 
        trace_split(df,'unknown')


    df_line = []
    max_len = 100
    for num,f in enumerate(os.listdir(folder)):
        if 'sess' in f:
            fn = os.path.join(folder,f)
            df = pickle.load(open(fn,'rb' ))
            data = df.tbs.values.copy()
            data.resize(max_len)
            row = pd.Series(data*100) #Mb/s
            df_line.append(row)

    df_unsupervised = pd.DataFrame(df_line)
    filename = 'use_case_files/df_'+cell+'_'+time_start+'_'+time_end+'.pkl'

    df_unsupervised.to_pickle(filename)
