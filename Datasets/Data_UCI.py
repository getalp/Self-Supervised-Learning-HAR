#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment if running on googlecolab 
# !pip install hickle
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd drive/MyDrive/PerCom2021-FL-master/


# In[ ]:


import numpy as np
import os
import pandas as pd
from subprocess import call
import requests 
np.random.seed(0)
import urllib.request
import zipfile
import hickle as hkl 
import tensorflow as tf


# In[ ]:


# functions for loading and downloading the dataset

# load a single file as a numpy array
def load_file(filepath):
	dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = np.dstack(loaded)
	return loaded
 
# load a dataset group, such as train or test
def load_dataset(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	filenames = list()
	# body acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# Framing data by windows
def segmentData(accData,time_step,step):
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])
    return segmentAccData

# download function for datasets
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[ ]:


# download and unzipping dataset
os.makedirs('dataset',exist_ok=True)
print("downloading...")            
data_directory = os.path.abspath("dataset/UCI HAR Dataset.zip")
if not os.path.exists(data_directory):
    download_url("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip",data_directory)
    print("download done")
else:
    print("dataset already downloaded")
    
data_directory2 = os.path.abspath("dataset/UCI HAR Dataset")
if not os.path.exists(data_directory2): 
    print("extracting data")
    with zipfile.ZipFile(data_directory, 'r') as zip_ref:
        zip_ref.extractall(os.path.abspath("dataset/"))
    print("data extracted in " + data_directory2)
else:
    print("Data already extracted in " + data_directory2)


# In[ ]:


trainSubjectList = pd.read_csv('dataset/UCI HAR Dataset/train/subject_train.txt', header=None, delim_whitespace=True).values
testSubjectList = pd.read_csv('dataset/UCI HAR Dataset/test/subject_test.txt', header=None, delim_whitespace=True).values

# 	return dataframe.values


# In[ ]:


# load all train
trainX, trainy = load_dataset('train', 'dataset/UCI HAR Dataset/')
trainy = np.asarray([x - 1 for x in trainy])

# load all test
testX, testy = load_dataset('test', 'dataset/UCI HAR Dataset/')
testy = np.asarray([x - 1 for x in testy])


# In[ ]:


# Combining datasets
datasets = list()
for x in range(0,trainX.shape[2]):
    datasets.append(np.concatenate((trainX[:,:,x],testX[:,:,x]), axis = 0))
datasets = np.dstack(datasets)


# In[ ]:


subjectList = np.vstack((trainSubjectList,testSubjectList)).squeeze()
labels = np.vstack((trainy,testy)).squeeze()


# In[ ]:


filteredLabels = np.asarray(labels)
filteredSubjectList = np.asarray(subjectList)
filteredData = np.asarray(datasets)


# In[ ]:


# filteredLabelsIndex = np.where(labels != 5)[0]
# filteredLabels = tf.gather(labels,filteredLabelsIndex).numpy()
# filteredSubjectList = tf.gather(subjectList,filteredLabelsIndex).numpy()
# filteredData = tf.gather(datasets,filteredLabelsIndex).numpy()


# In[ ]:


nbOfSubjects = len(np.unique(filteredSubjectList))
filteredSubjectList = filteredSubjectList - 1


# In[ ]:


subjectDataDict = {new_list: [] for new_list in range(nbOfSubjects)}
subjectLabelDict = {new_list: [] for new_list in range(nbOfSubjects)}


# In[ ]:


meanAcc = np.mean(filteredData[:,:,:3])
stdAcc = np.std(filteredData[:,:,:3])
varAcc = np.var(filteredData[:,:,:3])

meanGyro = np.mean(filteredData[:,:,3:])
stdGyro = np.std(filteredData[:,:,3:])
varGyro = np.var(filteredData[:,:,3:])

normalizedAllAcc = (filteredData[:,:,:3] - meanAcc) / stdAcc
normalizedAllGyro = (filteredData[:,:,3:] - meanGyro) / stdGyro
normalizedAll = np.dstack((normalizedAllAcc,normalizedAllGyro))


# In[ ]:


labelMapping = [5,1,0,3,4,6]
alignedLabels = np.asarray([labelMapping[label] for label in filteredLabels])


# In[ ]:


for index, subjectID in enumerate(filteredSubjectList):
    subjectDataDict[subjectID].append(normalizedAll[index])
    subjectLabelDict[subjectID].append(alignedLabels[index])


# In[ ]:


finalData = np.asarray([np.asarray(subjectDataDict[i]) for i in range(nbOfSubjects)], dtype=object)
finalLabel = np.asarray([np.asarray(subjectLabelDict[i]) for i in range(nbOfSubjects)], dtype=object)


# In[ ]:


def oneHot(label):
    onehot_label = tf.one_hot(
    label,
    10,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
    )
    return onehot_label

finalLabel = np.asarray([oneHot(clientLabel) for clientLabel in finalLabel],dtype=object)


# In[ ]:


dataName = 'UCI'
os.makedirs('datasetClientsUnion/'+dataName, exist_ok=True)
hkl.dump(finalData,'datasetClientsUnion/'+dataName+ '/clientsData.hkl' )
hkl.dump(finalLabel,'datasetClientsUnion/'+dataName+ '/clientsLabel.hkl' )


# In[ ]:




