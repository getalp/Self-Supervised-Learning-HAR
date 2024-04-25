#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import pandas as pd
import zipfile
from sklearn.preprocessing import StandardScaler
import hickle as hkl 
import requests 
import urllib.request
from scipy import signal
import tensorflow as tf
import resampy
np.random.seed(0)


# In[ ]:


# fomating data to adjust for dataset sensor errors
def formatData(data,dim):
    remainders = data.shape[0]%dim
    max_index = data.shape[0] - remainders
    data = data[:max_index,:]
    new = np.reshape(data, (-1, 128,3))
    return new

# segment data into windows
def segmentData(accData,time_step,step):
#     print(accData.shape)
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])
    return np.asarray(segmentAccData)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=0,usecols=[2,3,4])
    return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, filepath='',trainOrEval=0):
    loaded = list()
    for name in filenames:
        data = load_file(filepath + name)
        data = np.asarray(data)
#         print(data.shape)
#         data = segmentData(data,128,64)
        data = np.asarray(data)
        loaded.append(data)
    return loaded

# check if file exist
def isReadableFile(file_path, file_name,flag):
    full_path = file_path + "/" + file_name
    try:
        if not os.path.exists(file_path):
            return False
        elif not os.path.isfile(full_path):
            return False
        elif not os.access(full_path, os.R_OK):
            return False
        else:
            return True
    except IOError as ex:
        print ("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    except Error as ex:
        print ("Error({0}): {1}".format(ex.errno, ex.strerror))
    return False


# stairs down 0 
# stairs Up   1
# jumping     2
# lying       3
# standing    4 
# sitting     5
# running/jogging 6
# Walking     7

# load a dataset group
def load_dataset(group, datasetName='',activity='',orientation='',trainOrEval=0,client = 0):
    filepath = 'dataset/'+datasetName +'/'+ group + '/'
    filenames = list()
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv',0)):
        filenames += [str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv']
    X = load_group(filenames, filepath,trainOrEval)
    return X

# download function for datasets
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[ ]:


# definign activities and orientations of REALWORLD dataset
# activities = ['climbingdown','climbingup','jumping','lying','running','sitting','standing','walking'] 
activities = ['climbingdown','climbingup','running','sitting','standing','walking','lying','jumping']
# activities = ['standing','sitting','walking','climbingup','climbingdown'] 

# orientations = ['chest','forearm','head','shin','thigh','upperarm','waist']
orientations = ['waist']


# In[ ]:


orientationKeyMap = dict() 
for index,value in enumerate(orientations):
    orientationKeyMap[value] = index


# In[ ]:


# download and unzipping dataset
os.makedirs('dataset',exist_ok=True)
print("downloading...")            
data_directory = os.path.abspath("dataset/realworld2016_dataset.zip")
if not os.path.exists(data_directory):
    download_url("http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",data_directory)
    print("download done")
else:
    print("dataset already downloaded")
    
data_directory2 = os.path.abspath("dataset/realworld2016_dataset")
if not os.path.exists(data_directory2): 
    print("extracting data")
    with zipfile.ZipFile(data_directory, 'r') as zip_ref:
        zip_ref.extractall(os.path.abspath(data_directory2))
    print("data extracted in " + data_directory2)
else:
    print("Data already extracted in " + data_directory2)


# In[ ]:


def downSampleLowPass(toDownSampleData,factor):
    accX = signal.decimate(toDownSampleData[:,0],factor)
    accY = signal.decimate(toDownSampleData[:,1],factor)
    accZ = signal.decimate(toDownSampleData[:,2],factor)
    return np.dstack((accX,accY,accZ)).squeeze()


# In[ ]:


# Reading and processing all data
xAccListClient = list()
xGyrListClient = list()
yListClient = list()
clientsOrientation = []
nbAnomalies = 0 
for k in range(1,16):
    xAccList = list()
    xGyrList = list()
    yList = list()
    startingIndex = 0
    clientOrientation = []
    for activity in activities:
        for orientation in orientations:
            tempAcc = load_dataset('acc','REALWORLD',activity,orientation,0,k)
            tempGyro = load_dataset('Gyroscope','REALWORLD',activity,orientation,0,k)
            orientationLength = 0 
            for i in range(0, len(tempAcc)):  
                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])
                difference = accDataLength - gyroDataLength
                differenceAbs = abs(difference)
                
                differenceGyro = gyroDataLength - accDataLength
                if(differenceGyro > 1000):
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of:" +str(differenceAbs) + " found")
                    print("Acc data: "+str(accDataLength))
                    print("Gyro data: "+str(gyroDataLength))
                    tempGyro[i] = resampy.resample(tempGyro[i], gyroDataLength, accDataLength,axis = 0)
                    
                tempAcc[i] = segmentData(tempAcc[i],128,64)
                tempGyro[i] = segmentData(tempGyro[i],128,64)

                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])

                difference = accDataLength - gyroDataLength
                differenceAbs = abs(difference)
                if(differenceAbs < 21):
                    toAddShape = 0
                    if(difference > 0):
                        maxIndex = accDataLength-differenceAbs
                        xAccList.append(tempAcc[i][:maxIndex,:])
                        xGyrList.append(tempGyro[i])
                        toAddShape = tempGyro[i].shape[0]
                        yList.append(np.full((toAddShape), activities.index(activity)))   
                    else:
                        maxIndex = gyroDataLength-differenceAbs
                        xAccList.append(tempAcc[i])
                        xGyrList.append(tempGyro[i][:maxIndex,:])
                        toAddShape = tempAcc[i].shape[0]
                        yList.append(np.full((toAddShape), activities.index(activity)))
                    clientOrientation.append(np.full(toAddShape,orientationKeyMap[orientation]))
#                     startingIndex += toAddShape
    clientsOrientation.append(np.hstack((clientOrientation)))
    xAccListClient.append(np.vstack((xAccList)))
    xGyrListClient.append(np.vstack((xGyrList)))
    yListClient.append(np.hstack((yList)))


# In[ ]:


# # clientsOrientation
# dataName = 'RealWorld_Aligned'
# os.makedirs('datasetStandardized/'+dataName, exist_ok=True)
# hkl.dump(clientsOrientation,'datasetStandardized/'+dataName+ '/clientsOrientationRW.hkl')


# In[ ]:


# conversion to numpy array
xAccListClient = np.asarray(xAccListClient,dtype=object)
xGyrListClient = np.asarray(xGyrListClient,dtype=object)
yListClient = np.asarray(yListClient,dtype=object)


# In[ ]:


# stacking all partcipants client
allAcc = np.vstack((xAccListClient))
allGyro = np.vstack((xGyrListClient))


# In[ ]:


# Calculating features
meanAcc = np.mean(allAcc)
stdAcc = np.std(allAcc)
varAcc = np.var(allAcc)
stackedFeaturesAcc = np.hstack((meanAcc,stdAcc,varAcc))

meanGyro = np.mean(allGyro)
stdGyro = np.std(allGyro)
varGyro = np.var(allGyro)
stackedFeaturesGyro = np.hstack((meanGyro,stdGyro,varGyro))
stackedFeatures = np.vstack((stackedFeaturesAcc,stackedFeaturesGyro))


# In[ ]:


# channel-wise z-normalization
normalizedAcc = (xAccListClient - meanAcc)/stdAcc
normalizedGyro = (xGyrListClient - meanGyro)/stdGyro


# In[ ]:


alignedYListClient = []
for clientLabel in yListClient:
    alignedClientLabel = np.asarray([9 if labelInstance == 7  else labelInstance for labelInstance in clientLabel],dtype=object)
    alignedYListClient.append(alignedClientLabel)
alignedYListClient = np.asarray(alignedYListClient,dtype=object)


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

yListClient = [oneHot(clientLabel) for clientLabel in alignedYListClient]


# In[ ]:


subjectData = np.asarray([np.dstack((normAcc,normGyro))for normAcc,normGyro in zip(normalizedAcc,normalizedGyro)],dtype=object)


# In[ ]:


dataName = 'RealWorld_Waist'
os.makedirs('datasetClientsUnion/'+dataName, exist_ok=True)
hkl.dump(subjectData,'datasetClientsUnion/'+dataName+ '/clientsData.hkl' )
hkl.dump(np.asarray(yListClient),'datasetClientsUnion/'+dataName+ '/clientsLabel.hkl' )


# In[ ]:




