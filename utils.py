#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import hickle as hkl 
import tensorflow as tf
import seaborn as sns
import logging
from tensorflow.python.client import device_lib
import sklearn.metrics


import data2vec_model
import mae_model
import simclr_model

def create_classification_model_from_base_model(base_model, output_shape, model_name,dropout_rate = 0.3):
    intermediate_x = base_model.output
    x = tf.keras.layers.Dense(1024, activation=tf.nn.swish)(intermediate_x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

def oneHotSizeAdjuster(oneHotLabels,output_shape):
    argmaxsLabels = np.argmax(oneHotLabels,axis = -1)
    for newLabel,oldLabel in enumerate(np.unique(argmaxsLabels)):
        argmaxsLabels[argmaxsLabels == oldLabel ] = newLabel
    return tf.one_hot(argmaxsLabels,output_shape)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def projectTSNEWithShape(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels,globalPrototypesIndex = None ):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        style = labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=90,
        alpha=1.0,
        rasterized=True,
        markers = True)
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if(globalPrototypesIndex != None):
        plt.scatter(tsne_projections[globalPrototypesIndex,0],tsne_projections[globalPrototypesIndex,1], s=400,linewidth=3, facecolors='none', edgecolor='black')
    plt.savefig(filepath+fileName+".png", bbox_inches="tight", )
    plt.show()
    plt.clf()

def getF1Macro(groundTruth,predictions):
    truth_argmax = np.argmax(groundTruth, axis=1)
    pred_argmax = np.argmax(predictions, axis=1)
    return round(sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro'),4) * 100 


def projectTSNE(fileName,filepath,ACTIVITY_LABEL,labels_argmax,tsne_projections,unique_labels,globalPrototypesIndex = None ):
    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    graph = sns.scatterplot(
        x=tsne_projections[:,0], y=tsne_projections[:,1],
        hue=labels_argmax,
        palette=sns.color_palette(n_colors = len(unique_labels)),
        s=90,
        alpha=1.0,
        rasterized=True
    )
    legend = graph.legend_
    for j, label in enumerate(unique_labels):
        legend.get_texts()[j].set_text(ACTIVITY_LABEL[int(label)]) 
        

    plt.tick_params(
    axis='both',         
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,        
    labelbottom=False)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    if(globalPrototypesIndex != None):
        plt.scatter(tsne_projections[globalPrototypesIndex,0],tsne_projections[globalPrototypesIndex,1], s=400,linewidth=3, facecolors='none', edgecolor='black')
    plt.savefig(filepath+fileName+".png", bbox_inches="tight")
    plt.show()
    plt.clf()
def projectTSNEWithPosition(dataSetName,fileName,filepath,ACTIVITY_LABEL,labels_argmax,orientationsNames,clientOrientationTest,tsne_projections,unique_labels):
    classData = [ACTIVITY_LABEL[i] for i in labels_argmax]
    orientationData = [orientationsNames[i] for i in np.hstack((clientOrientationTest))]
    if(dataSetName == 'REALWORLD_CLIENT'):
        orientationName = 'Position'
    else:
        orientationName = 'Device'
    pandaData = {'col1': tsne_projections[:,0], 'col2': tsne_projections[:,1],'Classes':classData, orientationName :orientationData}
    pandaDataFrame = pd.DataFrame(data=pandaData)

    plt.figure(figsize=(16,16))
#     plt.title('HART Embeddings T-SNE')
    sns.scatterplot(data=pandaDataFrame, x="col1", y="col2", hue="Classes", style=orientationName,
                    palette=sns.color_palette(n_colors = len(unique_labels)),
                    s=90, alpha=1.0,rasterized=True,)
    plt.tick_params(
    axis='both',          
    which='both',     
    bottom=False,     
    top=False,         
    labelleft=False,       
    labelbottom=False)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    plt.savefig(filepath+fileName+".png", bbox_inches="tight")
    plt.show()
    plt.clf()

def plot_learningCurve(history, epochs, filepath, title):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'],markevery=[np.argmax(history.history['val_accuracy'])], ls="", marker="o",color="orange")
    plt.plot(epoch_range, history.history['accuracy'],markevery=[np.argmax(history.history['accuracy'])], ls="", marker="o",color="blue")

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.savefig(filepath+title+"LearningAccuracy.png", bbox_inches="tight")
    plt.show()
    plt.clf()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.plot(epoch_range, history.history['loss'],markevery=[np.argmin(history.history['loss'])], ls="", marker="o",color="blue")
    plt.plot(epoch_range, history.history['val_loss'],markevery=[np.argmin(history.history['val_loss'])], ls="", marker="o",color="orange")
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(filepath+title+"LearningLoss.png", bbox_inches="tight")
    plt.show()
    plt.clf()


def roundNumber(toRoundNb):
    return round(toRoundNb, 4) * 100
def converTensor(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

def extract_intermediate_model_from_base_model(base_model, intermediate_layer=7):
    model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.layers[intermediate_layer-1].output, name=base_model.name + "_layer_" + str(intermediate_layer))
    return model

def loadFineTuneData(trainingSamples,testingDataset,dataDir):
    fineTuneData = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingSamples)+'_samples_data.hkl')
    fineTuneLabel = hkl.load(dataDir + 'fineTuneData/'+testingDataset+'_'+str(trainingSamples)+'_samples_label.hkl')
    if(trainingSamples == 'all'):
        fineTuneData = np.vstack((fineTuneData))
        fineTuneLabel = np.vstack((fineTuneLabel))
    return fineTuneData,fineTuneLabel


def generatePatchedGraph(patchedSignals,fileName,lowerBound,upperBound,patch_count):
    for i, patch in enumerate(patchedSignals):
        ax = plt.gca()
        ax.set_ylim([lowerBound, upperBound])
        ax = plt.subplot(1, patch_count, i + 1)
        plt.ylim([lowerBound, upperBound])
        plt.plot(patch)
        plt.tick_params(
        axis='both',        
        which='both',      
        labelleft = False,
        left = False,
        bottom=False,      
        top=False,         
        labelbottom=False) 
    plt.savefig(fileName, bbox_inches="tight")
    plt.clf()

def loadPretrainedModel(method,architecture,leftOutDataset, returnType, activityCount = 10,modelDirectory = './'):
    pretrain_callbacks = []
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
            sensorWiseFramer = data2vec_model.SensorWiseFrameLayer(16,16)
            sensorWiseMaskEncoder = data2vec_model.SensorWiseMaskEncoder(enc_embedding_size,0.50,16)
            delta = 0.5
            decay = 0.9999
        elif(architecture == "ISPL"):        
            enc_embedding_size = 256
            teacherEncoder = data2vec_model.ispl_inception_teacher_encoder(enc_embedding_size)
            studentEncoder = data2vec_model.ispl_inception_encoder(enc_embedding_size)
            
            sensorWiseFramer = data2vec_model.FrameLayer(16,16)
            sensorWiseMaskEncoder = data2vec_model.MaskEncoder(enc_embedding_size,0.75,16)
            delta = 0.5
            decay = 0.998
        else:
            raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART")
            
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
            patch_layer = mae_model.SensorWiseFrameLayer(16,16)
            patch_encoder = mae_model.SensorWisePatchEncoder(16,enc_embedding_size,0.6)    
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
        
            patch_layer = mae_model.PatchLayer(16,16)
            patch_encoder = mae_model.PatchEncoder(16,enc_embedding_size,0.6)    
            mae_encoder = mae_model.ispl_inception_encoder(enc_embedding_size)
            mae_decoder = mae_model.ispl_inception_decoder(enc_embedding_size,
                                                           patch_count = patch_count,
                                                           filters_number = 192,
                                                           network_depth = 4,
                                                           output_shape = (128,6))
        else:
            raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART")
        pretrain_pipeline = mae_model.MaskedAutoencoder(patch_layer,
                                patch_encoder,
                                mae_encoder,
                                mae_decoder)
    
        SSL_loss = tf.keras.losses.MeanSquaredError()
        
    elif(method == 'SimCLR'):
        transform_funcs = []
        if(architecture == "HART"):
            encoder = simclr_model.HART_encoder((128,6))
            transform_funcs.append(simclr_model.noise_transform_vectorized)
    
        elif(architecture == "ISPL"):
            encoder = simclr_model.ispl_inception_encoder((128,6))
            transform_funcs.append(simclr_model.rotation_transform_vectorized)
    
        else:
            raise Exception("Unrecognized architecture, Please select one of the following: ISPL, HART")
        projection_heads = simclr_model.projection_head(encoder.output.shape[1])       
        transformations = simclr_model.generate_composite_transform_function_simple(transform_funcs)    
        pretrain_pipeline = simclr_model.SimCLR(encoder,
                            projection_heads,
                            transformations)
    #     Custom loss already defined inside of training pipeline
        SSL_loss = simclr_model.NT_Xent_loss(temperature = 0.1)
    
    else:
        raise Exception("Unrecognized algorithm, Please select one of the following: SimCLR, Data2vec, MAE")
    pretrain_pipeline.build(input_shape = (None,128,6))
    pretrain_pipeline.load_weights(modelDirectory+architecture+"_"+method+"_"+leftOutDataset+".h5")

    if(returnType == "pipeline"):
        return pretrain_pipeline
    elif(returnType == "featureExtractor"):
        return pretrain_pipeline.return_feature_extrator()
    elif(returnType == "classificationModel"):
        pretrained_FE = pretrain_pipeline.return_feature_extrator()
        classification_model = create_classification_model_from_base_model(pretrained_FE,activityCount,model_name = "pretrain_pipeline_classifier")
        return classification_model
    else:
        raise Exception("Unrecognized return state, Please select one of the following: pipeline, featureExtractor,classificationModel")
