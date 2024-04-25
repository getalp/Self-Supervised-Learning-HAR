#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import hickle as hkl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold
import copy
import csv
import __main__ as main
import argparse
import pandas as pd
from tabulate import tabulate
import time

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


# In[ ]:


# Library scripts
import utils 
import training
import data2vec_model
import mae_model
import simclr_model


# In[ ]:


experimentSetting = 'LODO'
# 'LOGO','LODO'
testingDataset = 'MotionSense'
# 'HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP'
evaluationType = 'group'
# 'subject','group'

method = 'MAE'
# Data2vec, MAE, SimCLR

architecture = 'HART'

finetune_epoch = 100

finetune_batch_size = 64

SSL_batch_size = 128

loss = 'Adam'
# 'LARS', 'Adam', 'SGD'

SSL_LR = 3e-4

FT_LR = 3e-4

input_shape = (128,6)

frame_length = 16

SSL_epochs = 200

masking_ratio = 75e-2

instance_number = 0

randomRuns = 5


# In[ ]:


datasets = ['HHAR','MobiAct','MotionSense','RealWorld_Waist','UCI','PAMAP']


# In[ ]:


architectures = ['HART','ISPL']


# In[ ]:


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--experimentSetting', type=str, default=experimentSetting, 
        help='Leave one dataset out or Leave one group out')  
    parser.add_argument('--testingDataset', type=str, default=testingDataset, 
        help='Left out dataset')  
    parser.add_argument('--evaluationType', type=str, default=evaluationType, 
        help='Dataset group evaluation or subject by subject evaluation')  
    parser.add_argument('--SSL_epochs', type=int, default=SSL_epochs, 
        help='SSL Epochs')  
    parser.add_argument('--SSL_batch_size', type=int, default=SSL_batch_size, 
        help='SSL batch_size')  
    parser.add_argument('--finetune_epoch', type=int, default=finetune_epoch, 
        help='Fine_tune Epochs')  
    parser.add_argument('--loss', type=str, default=loss, 
        help='Specify the loss') 
    parser.add_argument('--SSL_LR', type=float, default=SSL_LR, 
        help='Specify the learning rate for the SSL techniques') 
    parser.add_argument('--masking_ratio', type=float, default=masking_ratio, 
        help='Specify the masking ratio') 
    parser.add_argument('--frame_length', type=int, default=frame_length, 
        help='Specify the masking ratio') 
    parser.add_argument('--architecture', type=str, default=architecture, 
        help='Specify the architecture of the model to train with') 
    parser.add_argument('--method', type=str, default=method, 
        help='Specify the SSL method') 
    parser.add_argument('--instance_number', type=int, default=instance_number, 
        help='Specify the SSL method') 
    args = parser.parse_args()
    return args
def is_interactive():
    return not hasattr(main, '__file__')


# In[ ]:


tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[ ]:


if(input_shape[0] % frame_length != 0 ):
    raise Exception("Invalid segment size")
else:
    patch_count = input_shape[0] // frame_length
print("Number of segments : "+str(patch_count))


# In[ ]:


rootdir = './'
if not is_interactive():
    args = add_fit_args(argparse.ArgumentParser(description='SSL Pretraining Pipeline'))
    experimentSetting = args.experimentSetting
    testingDataset = args.testingDataset
    evaluationType = args.evaluationType
    SSL_epochs = args.SSL_epochs
    frame_length = args.frame_length
    SSL_batch_size = args.SSL_batch_size
    finetune_epoch = args.finetune_epoch
    loss = args.loss
    SSL_LR = args.SSL_LR
    masking_ratio = args.masking_ratio
    architecture = args.architecture
    method = args.method
    instance_number = args.instance_number


# In[ ]:


# # Sleep needed when launching jobs in parallel
# time.sleep((instance_number % 30) * 20 ) 
# # remove this before public release
# datasetIndex = (instance_number // (len(datasets) - 1)) % 6
# architectureIndex = instance_number // 30
# instance_number = instance_number%5
# testingDataset = datasets[datasetIndex]
# architecture = architectures[architectureIndex]
# print("instance_number: " +str(instance_number))
# print("testingDataset: " +str(datasets[datasetIndex]) + " architecture: " +str(architectures[architectureIndex]))


# In[ ]:


dataDir = rootdir+'Datasets/SSL_PipelineUnionV2/'+experimentSetting+'/'
# projectName = str(architecture)+'_Data2Vec_LayerNorm_mask_'+str(masking_ratio)+'_frameLength_'+str(frame_length)+'_SSL_epochs_'+str(SSL_epochs)
projectName = str(method) +"_"+str (architecture) + "_SSL_Epochs" + str(SSL_epochs)
testMode = False
if(finetune_epoch < 10):
    testMode = True
    projectName= projectName + '/tests'
    
dataSetting = testingDataset

project_directory = rootdir+'results/'+projectName+'/'+str(instance_number)+'/'
working_directory = project_directory+dataSetting+'/'
pretrained_dir = working_directory + evaluationType + '/'
    
initWeightDir_pretrain = project_directory+'ini_'+str(method)+'_'+str(architecture)+'_Pretraining_Weights.h5'
val_checkpoint_pipeline_weights = working_directory+"best_val_"+str(method)+"_pretrain.h5"
trained_pipeline_weights = working_directory+"trained_"+str(method)+"_pretrain.h5"
random_FT_weights = working_directory+"ini_"+str(method)+"_HART_Classification_Weights.h5"    
trained_FT_weights = working_directory+"trained_"+str(method)+"_dowmstream.h5"

os.makedirs(pretrained_dir, exist_ok=True)


# In[ ]:


datasetList = ["HHAR","MobiAct","MotionSense","RealWorld_Waist","UCI","PAMAP"] 


# In[ ]:


SSLdatasetList = copy.deepcopy(datasetList)
SSLdatasetList.remove(testingDataset)
SSL_data = []
SSL_label = []

SSL_val_data = []
SSL_val_label = []

for datasetName in SSLdatasetList:
    SSL_data.append(hkl.load(dataDir + 'testData/'+str(datasetName)+'_data.hkl'))
    SSL_data.append(hkl.load(dataDir + 'fineTuneData/'+str(datasetName)+'_all_samples_data.hkl'))
    SSL_val_data.append(hkl.load(dataDir + 'valData/'+str(datasetName)+'_data.hkl'))

SSL_data = np.vstack((np.hstack((SSL_data))))

SSL_val_data = np.vstack((np.hstack((SSL_val_data))))

testData = hkl.load(dataDir + 'testData/'+testingDataset+'_data.hkl')
testLabel = hkl.load(dataDir + 'testData/'+testingDataset+'_label.hkl')

valData = hkl.load(dataDir + 'valData/'+testingDataset+'_data.hkl')
valLabel = hkl.load(dataDir + 'valData/'+testingDataset+'_label.hkl')

testData = np.vstack((testData))
testLabel = np.vstack((testLabel))
valData = np.vstack((valData))
valLabel = np.vstack((valLabel))


# In[ ]:


# Here we are getting the labels presented only in the target dataset and calculating the suitable output shape.
ALL_ACTIVITY_LABEL = np.asarray(['Downstairs', 'Upstairs','Running','Sitting','Standing','Walking','Lying','Cycling','Nordic_Walking','Jumping'])
uniqueClassIDs = np.unique(np.argmax(testLabel,axis = -1))
ACTIVITY_LABEL = ALL_ACTIVITY_LABEL[uniqueClassIDs]
output_shape = len(ACTIVITY_LABEL)


# In[ ]:


pretrain_callbacks = []


# In[ ]:


# if(method == 'Data2vec'):
#     if(architecture == "HART"):
#         enc_embedding_size = 192
#         teacherEncoder = data2vec_model.HART_teacher_encoder(projection_dim = enc_embedding_size, num_heads = 3,
#                                                 filterAttentionHead = 4, 
#                                                 convKernels = [3, 7, 15, 31, 31, 31],
#                                                 layerAverage = 3)
        
#         studentEncoder = data2vec_model.HART_student_encoder(projection_dim = enc_embedding_size, num_heads = 3,
#                                                 filterAttentionHead = 4, 
#                                                 convKernels = [3, 7, 15, 31, 31, 31],)
        
        
#         sensorWiseFramer = data2vec_model.SensorWiseFrameLayer(frame_length,frame_length)
#         sensorWiseMaskEncoder = data2vec_model.SensorWiseMaskEncoder(enc_embedding_size,masking_ratio,frame_length)
#     elif(architecture == "ISPL"):        
#         enc_embedding_size = 256
#         teacherEncoder = data2vec_model.ispl_inception_teacher_encoder(enc_embedding_size)
#         studentEncoder = data2vec_model.ispl_inception_encoder(enc_embedding_size)
        
#         sensorWiseFramer = data2vec_model.FrameLayer(frame_length,frame_length)
#         # sensorWiseMaskEncoder = data2vec_model.MaskEncoder(enc_embedding_size,masking_ratio,frame_length)
#         sensorWiseMaskEncoder = data2vec_model.MaskEncoder(enc_embedding_size,masking_ratio,frame_length)

#     else:
#         raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
        
#     pretrain_pipeline =  data2vec_model.Data2Vec(sensorWiseFramer,
#                                          sensorWiseMaskEncoder,
#                                          teacherEncoder,
#                                          studentEncoder)   
#     SSL_loss = tf.keras.losses.Huber()
    
#     pretrain_callbacks.append(data2vec_model.EMA())
    
#     for teacherLayers in teacherEncoder.layers:
#         teacherLayers.trainable = False
# elif(method == 'MAE'):
#     if(architecture == "HART"):
#         enc_embedding_size = 192
#         patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
#         patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,True,masking_ratio)    
#         mae_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
#                                              num_heads = 3,
#                                              filterAttentionHead = 4, 
#                                              convKernels = [3, 7, 15, 31, 31, 31])

#         mae_decoder = mae_model.HART_decoder(enc_embedding_size = enc_embedding_size,
#                                              num_heads = 3,
#                                              filterAttentionHead = 4, 
#                                              convKernels = [3, 7, 7, 15])
#     elif(architecture == "ISPL"):
# #         this parameter was hardcoded into ISPL for MAE to align with original author without changing the architecture.
#         enc_embedding_size = 256
#         dec_embedding_size = (enc_embedding_size//2)
#         patch_layer = mae_model.PatchLayer(frame_length,frame_length)
#         patch_encoder = mae_model.PatchEncoder(frame_length,enc_embedding_size,True,masking_ratio)    
#         mae_encoder = mae_model.ispl_inception_encoder(enc_embedding_size)
#         mae_decoder = mae_model.ispl_inception_decoder(enc_embedding_size,patch_count = patch_count,output_shape = input_shape)
#     else:
#         raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
#     pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
#                             patch_encoder,
#                             mae_encoder,
#                             mae_decoder)

#     SSL_loss = tf.keras.losses.MeanSquaredError()
    
# elif(method == 'SimCLR'):
#     if(architecture == "HART"):
#         encoder = simclr_model.HART_encoder(input_shape)
#     elif(architecture == "ISPL"):
#         encoder = simclr_model.ispl_inception_encoder(input_shape)
#     else:
#         raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
#     transform_funcs = [
#         simclr_model.rotation_transform_vectorized # Use rotation trasnformation
#     ]    
#     projection_heads = simclr_model.projection_head(encoder.output.shape[1])
#     transformations = simclr_model.generate_composite_transform_function_simple(transform_funcs)
#     pretrain_pipeline = simclr_model.SimCLR(encoder,
#                         projection_heads,
#                         transformations)
    
# #     Custom loss already defined inside of training pipeline
#     SSL_loss = simclr_model.NT_Xent_loss(temperature = 0.01)
# else:
#     raise Exception("Unrecognized algorithm, Please select one of the following: SimCLR, Data2vec, MAE")
# # ISPL,HART,MobileHART,DeepConvLSTM


# In[ ]:


if(method == 'Data2vec'):
    if(architecture == "HART"):
        enc_embedding_size = 192
        teacherEncoder = data2vec_model.HART_teacher_encoder(projection_dim = enc_embedding_size, num_heads = 3,
                                                filterAttentionHead = 4, 
                                                convKernels = [3, 7, 15, 31, 31, 31],
                                                layerAverage = 3)
        studentEncoder = data2vec_model.HART_student_encoder(projection_dim = enc_embedding_size, num_heads = 3,
                                                filterAttentionHead = 4, 
                                                convKernels = [3, 7, 15, 31, 31, 31],)
        sensorWiseFramer = data2vec_model.SensorWiseFrameLayer(frame_length,frame_length)
        sensorWiseMaskEncoder = data2vec_model.SensorWiseMaskEncoder(enc_embedding_size,0.50,frame_length)
        delta = 0.5
        decay = 0.9999
    elif(architecture == "ISPL"):        
        enc_embedding_size = 256
        teacherEncoder = data2vec_model.ispl_inception_teacher_encoder(enc_embedding_size)
        studentEncoder = data2vec_model.ispl_inception_encoder(enc_embedding_size)
        
        sensorWiseFramer = data2vec_model.FrameLayer(frame_length,frame_length)
        sensorWiseMaskEncoder = data2vec_model.MaskEncoder(enc_embedding_size,0.75,frame_length)
        delta = 0.5
        decay = 0.998

    else:
        raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
        
    pretrain_pipeline =  data2vec_model.Data2Vec(sensorWiseFramer,
                                         sensorWiseMaskEncoder,
                                         teacherEncoder,
                                         studentEncoder)   
    SSL_loss = tf.keras.losses.Huber(delta = delta)
    
    pretrain_callbacks.append(data2vec_model.EMA(decay = decay))
    
    for teacherLayers in teacherEncoder.layers:
        teacherLayers.trainable = False
elif(method == 'MAE'):
    if(architecture == "HART"):
        enc_embedding_size = 192
        patch_layer = mae_model.SensorWiseFrameLayer(frame_length,frame_length)
        patch_encoder = mae_model.SensorWisePatchEncoder(frame_length,enc_embedding_size,0.6)    
        mae_encoder = mae_model.HART_encoder(enc_embedding_size,                                                     
                                             num_heads = 3,
                                             filterAttentionHead = 4, 
                                             convKernels = [3, 7, 15, 31, 31, 31])
    
        mae_decoder = mae_model.HART_decoder(enc_embedding_size = enc_embedding_size,
                                             projection_dim = 256,
                                             patch_count = patch_count,
                                             num_heads = 3,
                                             filterAttentionHead = 4, 
                                             convKernels = [3, 7, 15, 31, 31, 31])
    elif(architecture == "ISPL"):
        enc_embedding_size = 256
    
        patch_layer = mae_model.PatchLayer(frame_length,frame_length)
        patch_encoder = mae_model.PatchEncoder(frame_length,enc_embedding_size,0.6)    
        mae_encoder = mae_model.ispl_inception_encoder(enc_embedding_size)
        mae_decoder = mae_model.ispl_inception_decoder(enc_embedding_size,
                                                       patch_count = patch_count,
                                                       filters_number = 192,
                                                       network_depth = 4,
                                                       output_shape = input_shape)
    else:
        raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
    pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
                            patch_encoder,
                            mae_encoder,
                            mae_decoder)

    SSL_loss = tf.keras.losses.MeanSquaredError()
    
elif(method == 'SimCLR'):
    transform_funcs = []
    if(architecture == "HART"):
        encoder = simclr_model.HART_encoder(input_shape)
        transform_funcs.append(simclr_model.noise_transform_vectorized)

    elif(architecture == "ISPL"):
        encoder = simclr_model.ispl_inception_encoder(input_shape)
        transform_funcs.append(simclr_model.rotation_transform_vectorized)

    else:
        raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART,HART_BASE")
    projection_heads = simclr_model.projection_head(encoder.output.shape[1])       
    transformations = simclr_model.generate_composite_transform_function_simple(transform_funcs)    
    pretrain_pipeline = simclr_model.SimCLR(encoder,
                        projection_heads,
                        transformations)
#     Custom loss already defined inside of training pipeline
    SSL_loss = simclr_model.NT_Xent_loss(temperature = 0.1)

else:
    raise Exception("Unrecognized algorithm, Please select one of the following: SimCLR, Data2vec, MAE")


# In[ ]:


optimizer = tf.keras.optimizers.Adam(SSL_LR)

pretrain_pipeline.compile(optimizer=optimizer, loss=SSL_loss, metrics=[])

# Forcing a build to the model 
pretrain_pipeline.build(input_shape = (None,128,6))

if(not os.path.exists(initWeightDir_pretrain)):
    print("Initialized model weights not found, generating one")
    pretrain_pipeline.save_weights(initWeightDir_pretrain)
else:
    pretrain_pipeline.load_weights(initWeightDir_pretrain)
    print("Initialized model weights loaded")


# In[ ]:


pretrained_FE = pretrain_pipeline.return_feature_extrator()
classification_model = utils.create_classification_model_from_base_model(pretrained_FE,output_shape,model_name = "pretrain_pipeline_classifier")
FE_Layers = len(pretrained_FE.layers) + 1
if(not os.path.exists(random_FT_weights)):
    classification_model.save_weights(random_FT_weights)


# In[ ]:


if(not os.path.exists(trained_pipeline_weights)):
    print(trained_pipeline_weights)
    print("Not Found")
    best_val_model_callback = tf.keras.callbacks.ModelCheckpoint(val_checkpoint_pipeline_weights,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=2)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    pretrain_callbacks.append(best_val_model_callback)
    pretrain_callbacks.append(stop_early)
    history = pretrain_pipeline.fit(SSL_data,
                                    validation_data = (SSL_val_data,SSL_val_data), 
                                    batch_size = SSL_batch_size, 
                                    epochs = SSL_epochs,
                                    callbacks=pretrain_callbacks,
                                    verbose=2)
    
    plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'], label = 'Train Loss')
    plt.plot(history.history['val_loss'], label = 'Val Loss')
    plt.plot(history.history['val_loss'],markevery=[np.argmin(history.history['val_loss'])], ls="", marker="o",color="orange")
    plt.plot(history.history['loss'],markevery=[np.argmin(history.history['loss'])], ls="", marker="o",color="blue")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(working_directory+"lossCurve.png", bbox_inches="tight")
    plt.clf()

    pretrain_pipeline.load_weights(val_checkpoint_pipeline_weights)
    pretrain_pipeline.save_weights(trained_pipeline_weights)
    classification_model.save_weights(trained_FT_weights)
    perplexity = 30.0
    embeddings = pretrain_pipeline.predict(testData, batch_size=1024,verbose=0)
    tsne_model = sklearn.manifold.TSNE(perplexity=perplexity, verbose=0, random_state=42)
    tsne_projections = tsne_model.fit_transform(embeddings)
    labels_argmax = np.argmax(testLabel, axis=1)
    unique_labels = np.unique(labels_argmax)
    utils.projectTSNE('TSNE_Embeds',pretrained_dir,ALL_ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
    utils.projectTSNEWithShape('TSNE_Embeds_shape',pretrained_dir,ALL_ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels )
    hkl.dump(tsne_projections,pretrained_dir+'tsne_projections.hkl')
else:
    pretrain_pipeline.load_weights(trained_pipeline_weights)
    classification_model.load_weights(trained_FT_weights)
    print("Pre-trained model found, skipping training of pretrained model",flush = True)


# ### Downstream tasks

# In[ ]:


samples = ['1','2', '5', '10', '25', '50', '100','all']


# In[ ]:


def oneHotSizeAdjuster(oneHotLabels):
    argmaxsLabels = np.argmax(oneHotLabels,axis = -1)
    for newLabel,oldLabel in enumerate(np.unique(argmaxsLabels)):
        argmaxsLabels[argmaxsLabels == oldLabel ] = newLabel
    return tf.one_hot(argmaxsLabels,output_shape)


# In[ ]:


valLabel = utils.oneHotSizeAdjuster(valLabel,output_shape)
testLabel = utils.oneHotSizeAdjuster(testLabel,output_shape)


# In[ ]:


sampleResult = {}
for sampleCount in samples:
    print("Now downstreaming on "+testingDataset+" dataset with samples count: "+str(sampleCount), flush = True)
    evaluation_dir = pretrained_dir+'samples_'+ str(sampleCount) + '/'
    os.makedirs(evaluation_dir, exist_ok=True)
    fineTuneData,fineTuneLabel = utils.loadFineTuneData(sampleCount,testingDataset,dataDir)
    fineTuneLabel = utils.oneHotSizeAdjuster(fineTuneLabel,output_shape)
    evaluationsF1 = training.downStreamPipeline(fineTuneData,fineTuneLabel,valData,valLabel,testData,testLabel,
                                                evaluation_dir,
                                                classification_model,
                                                FE_Layers,random_FT_weights,trained_FT_weights,
                                                finetune_epoch = finetune_epoch, 
                                                finetune_batch_size = finetune_batch_size)
    sampleResult['sample_'+str(sampleCount)] = evaluationsF1


# In[ ]:


npRatio = np.asarray(list(sampleResult.values())).T
evaluationMethods = ['Result_Frozen_FE','Result_Unfrozen_FE']
for evalIndex, methods in enumerate(evaluationMethods):
    toWriteEvaluation = {}
    toWriteEvaluation['dataset'] = [testingDataset]
    for ratioIndex, sample in enumerate(samples):
        toWriteEvaluation['Sample_'+str(sample)] = [npRatio[evalIndex][ratioIndex]]
    tabular = tabulate(toWriteEvaluation, headers="keys")
    print(methods)
    print(tabular)
    print()
    text_file = open(pretrained_dir +methods+'_report.csv',"w")
    text_file.write(tabular)
    text_file.close()


# In[ ]:


allTrained = True 
for methods in evaluationMethods:
    print("Processing "+str(methods) +" report")
    fullReport = []
    ratioHeaders = ['Sample_'+str(sample) for sample in samples]
    ratioHeaders.insert(0, "dataset")
    fullReport.append(ratioHeaders)
    if(allTrained):
        for datasetName in datasetList:
            checkDir = rootdir+'results/'+projectName+'/'+datasetName+'/'+evaluationType+'/'+methods+"_report.csv"
            if(not os.path.exists(checkDir)):
                print("Dir below not found:")
                print(checkDir)
                allTrained = False
                break
            readData = pd.read_table(checkDir, delim_whitespace=True)
            fullReport.append(readData.to_numpy()[1])
    else:
        break
    if(allTrained):
        print("Generating "+str(methods) + " report")
        tabular2 = tabulate(fullReport)
        text_file = open(rootdir+'results/'+projectName+'/'+str(evaluationType)+'_'+str(methods)+'_report.csv',"w")
        text_file.write(tabular2)
        text_file.close()
        print(tabular2)


# In[ ]:


formattedResults = np.asarray(list(sampleResult.values())).T


# In[ ]:


hkl.dump(formattedResults,working_directory+'training_results.hkl' )


# In[ ]:




