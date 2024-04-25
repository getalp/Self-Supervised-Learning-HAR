#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment if running on googlecolab 
# !pip install hickle
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd drive/MyDrive/PerCom2021-FL-master/


# In[ ]:


import hickle as hkl 
import numpy as np
import os
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


# In[ ]:


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
randomSeed = 0
np.random.seed(randomSeed)


# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


mainDir = './Datasets'


# In[ ]:


datasetList = ['HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'] 


# In[ ]:


dirName =  mainDir + 'SSL_PipelineUnionV2/LODO'
os.makedirs(dirName, exist_ok=True)


# In[ ]:


fineTuneDir = 'fineTuneData'
testDir = 'testData'
valDir = 'valData'
datasetDir = 'datasets'
# os.makedirs(dirName+'/'+datasetDir, exist_ok=True)
os.makedirs(dirName+'/'+fineTuneDir, exist_ok=True)
os.makedirs(dirName+'/'+testDir, exist_ok=True)
os.makedirs(dirName+'/'+valDir, exist_ok=True)


# In[ ]:


fineTuneData = []
fineTuneLabel = []

for datasetIndex,dataSetName in enumerate(datasetList):
    datasetLabel = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsLabel.hkl')
    datasetTrain = hkl.load(mainDir + 'datasetClientsUnion/'+dataSetName+'/clientsData.hkl')
#     hkl.dump(datasetTrain,dirName+'/'+datasetDir+ '/'+dataSetName+'_data.hkl')
#     hkl.dump(datasetLabel,dirName+'/'+datasetDir+ '/'+dataSetName+'_label.hkl')
    trainingData = []
    testingData = []
    validatingData = []
    
    trainingLabel = []
    testingLabel = []
    validatingLabel = []
    
    for datasetData, datasetLabels in zip(datasetTrain,datasetLabel):
        nonSoftMaxedLabels = np.argmax(datasetLabels,axis = -1)
        
        skf = StratifiedKFold(n_splits=10,shuffle = False)
        skf.get_n_splits(datasetData, nonSoftMaxedLabels)
        partitionedData = list()
        partitionedLabel = list()
        testIndex = []
        
        for train_index, test_index in skf.split(datasetData, nonSoftMaxedLabels):
            testIndex.append(test_index)

        trainIndex = np.hstack((testIndex[:7]))
        devIndex = testIndex[8]
        testIndex = np.hstack((testIndex[8:]))

        X_train = tf.gather(datasetData,trainIndex).numpy()
        X_val = tf.gather(datasetData,devIndex).numpy()
        X_test = tf.gather(datasetData,testIndex).numpy()

        y_train = tf.gather(nonSoftMaxedLabels,trainIndex).numpy()
        y_val = tf.gather(nonSoftMaxedLabels,devIndex).numpy()
        y_test = tf.gather(nonSoftMaxedLabels,testIndex).numpy()
        
        y_train = tf.one_hot(y_train,10)
        y_val = tf.one_hot(y_val,10)
        y_test = tf.one_hot(y_test,10)

        trainingData.append(X_train)
        validatingData.append(X_val)
        testingData.append(X_test)
        
        trainingLabel.append(y_train)
        validatingLabel.append(y_val)
        testingLabel.append(y_test)
        
        
    # testingLabel = np.asarray(testingLabel)
    # testingData = np.asarray(testingData)
    
    # validatingData = np.asarray(validatingData)
    # validatingLabel = np.asarray(validatingLabel)

    # trainingLabel = np.asarray(trainingLabel)
    # trainingData = np.asarray(trainingData)
    
    fineTuneData.append(trainingData)
    fineTuneLabel.append(trainingLabel)

    hkl.dump(trainingData,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_all_samples_data.hkl')
    hkl.dump(trainingLabel,dirName+'/'+fineTuneDir+ '/'+dataSetName+'_all_samples_label.hkl')
    hkl.dump(testingData,dirName+'/'+testDir+ '/'+dataSetName+'_data.hkl' )
    hkl.dump(testingLabel,dirName+'/'+testDir+ '/'+dataSetName+'_label.hkl' )
    hkl.dump(validatingData,dirName+'/'+valDir+ '/'+dataSetName+'_data.hkl' )
    hkl.dump(validatingLabel,dirName+'/'+valDir+ '/'+dataSetName+'_label.hkl' )
# fineTuneData = np.asarray(fineTuneData)
# fineTuneLabel = np.asarray(fineTuneLabel)


dirName+'/'+fineTuneDir+ '/'+dataSetName+'_All_samples_data.hkl'


# In[ ]:


fineTuneSamples = [100, 50, 25, 10, 5, 2, 1]


# In[ ]:


# fineTuneData = np.vstack((np.hstack((fineTuneData))))
# fineTuneLabel = np.vstack((np.hstack((fineTuneLabel))))


# In[ ]:


gen = np.random.default_rng()

for index, (trainingDataSubject, traningLabelSubject) in enumerate(zip(fineTuneData,fineTuneLabel)):

    stackedData =  np.vstack(trainingDataSubject)
    stackedLabel = np.vstack(traningLabelSubject)
    stackedSoftMaxLabels = np.argmax(stackedLabel,axis = -1).ravel()
    uniqueLabels = np.unique(stackedSoftMaxLabels)
    
    datasetXSamples = {new_list: [] for new_list in fineTuneSamples}
    datasetYSamples = {new_list: [] for new_list in fineTuneSamples}

    for labels in uniqueLabels:
        labelLocation = np.where(stackedSoftMaxLabels == labels)[0]
        labelLocation = gen.choice(labelLocation, np.max(fineTuneSamples), replace=False)
        for sampleCount in fineTuneSamples:
            datasetXSamples[sampleCount].append(stackedData[labelLocation][:sampleCount])
            datasetYSamples[sampleCount].append(stackedLabel[labelLocation][:sampleCount])
    
    fileSavePath = dirName+'/'+fineTuneDir+ '/'
    os.makedirs(fileSavePath, exist_ok=True)
    for sampleCount in fineTuneSamples:
        hkl.dump(np.vstack((datasetXSamples[sampleCount])),fileSavePath + datasetList[index]+'_'+str(int(sampleCount))+'_samples_data.hkl')
        hkl.dump(np.vstack((datasetYSamples[sampleCount])),fileSavePath + datasetList[index]+'_'+str(int(sampleCount))+'_samples_label.hkl')

