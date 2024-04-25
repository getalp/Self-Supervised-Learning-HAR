#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import mobiact data
import scipy
import scipy.signal
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import stats
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import hickle as hkl


# In[ ]:


def standardize_data(deviceData):
    deviceDataAcc = deviceData[:,:,:3].astype(np.float32)
    deviceDataGyro = deviceData[:,:,3:].astype(np.float32)
    accMean =  np.mean(deviceDataAcc)
    accStd =  np.std(deviceDataAcc)
    gyroMean =  np.mean(deviceDataGyro)
    gyroStd =  np.std(deviceDataGyro)
    deviceDataAcc = (deviceDataAcc - accMean)/accStd
    deviceDataGyro = (deviceDataGyro - gyroMean)/gyroStd
    deviceData = np.dstack((deviceDataAcc,deviceDataGyro))
    return deviceData

def create_segments_and_labels_Mobiact_fixed(df, time_steps, step, label_name = "LabelsEncoded", n_features= 6):
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        acc_x = df['acc_x'].values[i: i + time_steps]
        acc_y = df['acc_y'].values[i: i + time_steps]
        acc_z = df['acc_z'].values[i: i + time_steps]

        gyro_x = df['gyro_x'].values[i: i + time_steps]
        gyro_y = df['gyro_y'].values[i: i + time_steps]
        gyro_z = df['gyro_z'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        label = stats.mode(df[label_name][i: i + time_steps])[0][0]
        reshaped = np.dstack([acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z]).reshape(time_steps, n_features)
        segments.append(reshaped)
        labels.append(label)
    return np.asarray(segments), np.asarray(labels)


# In[ ]:


base_dir = "dataset"
df_all_data = pd.read_csv(base_dir +'/Mob_data_01.csv')

unique_user_ids = df_all_data['user_id'].unique()
all_data = []
all_labels = []
for user_id in unique_user_ids:
    selected_data = df_all_data.loc[df_all_data['user_id'] == user_id]
    x, y = create_segments_and_labels_Mobiact_fixed(selected_data, 256, 128)
    x_aligned = scipy.signal.decimate(x, q = 2, n=None, ftype='iir', axis=1, zero_phase=True)
    mapping = [2,9,3,4,0,1,5]
    y_aligned =  np.hstack(mapping[labelIndex] for labelIndex in y)
    y_oneHot_aligned = tf.one_hot(y_aligned,10)
    all_data.append(x_aligned)
    all_labels.append(y_oneHot_aligned)


# In[ ]:


all_labels = np.asarray(all_labels,dtype=object)
all_data = np.asarray(all_data,dtype=object)


# In[ ]:


subjectIndex = []
for data in all_data:
    subjectIndex.append(data.shape[0])


# In[ ]:


allData = np.vstack((all_data))
standardizedData = standardize_data(allData)


# In[ ]:


subjectData = []
startIndex = 0
endIndex = 0
for index in subjectIndex:
    endIndex += index
    subjectData.append(standardizedData[startIndex:endIndex])
    startIndex = endIndex
subjectData = np.asarray(subjectData,dtype=object)


# In[ ]:


dataName = 'MobiAct'
os.makedirs('datasetClientsUnion/'+dataName, exist_ok=True)
hkl.dump(subjectData,'datasetClientsUnion/'+dataName+ '/clientsData.hkl' )
hkl.dump(all_labels,'datasetClientsUnion/'+dataName+ '/clientsLabel.hkl' )


# In[ ]:




