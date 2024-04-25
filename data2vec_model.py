#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
import numpy as np


class DropPath(layers.Layer):
    def __init__(self, drop_prob=0.0, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x,training=None):
        if(training):
            input_shape = tf.shape(x)
            batch_size = input_shape[0]
            rank = x.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
            path_mask = tf.floor(random_tensor)
            output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
            return output
        else:
            return x 

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'drop_prob': self.drop_prob,})
        return config

class SensorWiseMHA(layers.Layer):
    def __init__(self, projectionQuarter, num_heads,startIndex,stopIndex,dropout_rate = 0.0,dropPathRate = 0.0, **kwargs):
        super(SensorWiseMHA, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.MHA = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projectionQuarter, dropout = dropout_rate )
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.dropPathRate = dropPathRate
        self.DropPath = DropPath(dropPathRate)
    def call(self, inputData, training=None, return_attention_scores = False):
        extractedInput = inputData[:,:,self.startIndex:self.stopIndex]
        if(return_attention_scores):
            MHA_Outputs, attentionScores = self.MHA(extractedInput,extractedInput,return_attention_scores = True )
            return MHA_Outputs , attentionScores
        else:
            MHA_Outputs = self.MHA(extractedInput,extractedInput)
            MHA_Outputs = self.DropPath(MHA_Outputs)
            return MHA_Outputs
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'num_heads': self.num_heads,
            'startIndex': self.startIndex,
            'dropout_rate': self.dropout_rate,
            'stopIndex': self.stopIndex,
            'dropPathRate': self.dropPathRate,})
        return config

class liteFormer(layers.Layer):
    def __init__(self,startIndex,stopIndex, projectionSize, kernelSize = 16, attentionHead = 3, use_bias=False, dropPathRate = 0.0,dropout_rate = 0,**kwargs):
        super(liteFormer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.startIndex = startIndex
        self.stopIndex = stopIndex
        self.kernelSize = kernelSize
        self.softmax = tf.nn.softmax
        self.projectionSize = projectionSize
        self.attentionHead = attentionHead 
        self.dropPathRate = dropPathRate
        self.dropout_rate = dropout_rate
        self.DropPathLayer = DropPath(dropPathRate)
        self.projectionHalf = projectionSize // 2
    def build(self, input_shape):
        self.depthwise_kernel = [self.add_weight(
            shape=(self.kernelSize,1,1),
            initializer="glorot_uniform",
            trainable=True,
            name="convWeights"+str(_),
            dtype="float32") for _ in range(self.attentionHead)]
        if self.use_bias:
            self.convBias = self.add_weight(
                shape=(self.attentionHead,), 
                initializer="glorot_uniform", 
                trainable=True,  
                name="biasWeights",
                dtype="float32"
            )
        
    def call(self, inputs,training=None):
        formattedInputs = inputs[:,:,self.startIndex:self.stopIndex]
#         print(inputs.shape)
        inputShape = tf.shape(formattedInputs)
#         reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,self.projectionSize))
        reshapedInputs = tf.reshape(formattedInputs,(-1,self.attentionHead,inputShape[1]))
        if(training):
            for convIndex in range(self.attentionHead):
                self.depthwise_kernel[convIndex].assign(self.softmax(self.depthwise_kernel[convIndex], axis=0))
        convOutputs = [tf.nn.conv1d(
            reshapedInputs[:,convIndex:convIndex+1,:],
            self.depthwise_kernel[convIndex],
            stride = 1,
            padding = 'SAME',
            data_format='NCW',) for convIndex in range(self.attentionHead) ]
        convOutputs = tf.convert_to_tensor(convOutputs)
        convOutputs = self.DropPathLayer(convOutputs)

        shape = tf.shape(formattedInputs)
        localAttention = tf.reshape(convOutputs,shape)
        return localAttention
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'use_bias': self.use_bias,
            'patchCount': self.patchCount,
            'kernelSize': self.kernelSize,
            'startIndex': self.startIndex,
            'stopIndex': self.stopIndex,
            'projectionSize': self.projectionSize,
            'dropPathRate': self.dropPathRate,
            'dropout_rate': self.dropout_rate,
            'attentionHead': self.attentionHead,})
        return config          

class mixAccGyro(layers.Layer):
    def __init__(self,projectionQuarter,projectionHalf,projection_dim,**kwargs):
        super(mixAccGyro, self).__init__(**kwargs)
        self.projectionQuarter = projectionQuarter
        self.projectionHalf = projectionHalf
        self.projection_dim = projection_dim
        self.projectionThreeFourth = self.projectionHalf+self.projectionQuarter
        self.mixedAccGyroIndex = tf.reshape(tf.transpose(tf.stack(
            [np.arange(projectionQuarter,projectionHalf), np.arange(projectionHalf,projectionHalf + projectionQuarter)])),[-1])
        self.newArrangement = tf.concat((np.arange(0,projectionQuarter),self.mixedAccGyroIndex,np.arange(self.projectionThreeFourth,projection_dim)),axis = 0)
    def call(self, inputs):
        return tf.gather(inputs,self.newArrangement,axis= 2)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'projectionQuarter': self.projectionQuarter,
            'projectionHalf': self.projectionHalf,
            'projection_dim': self.projection_dim,
        })
        return config


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def mlp2(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0],activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    return x

def depthMLP(x, hidden_units, dropout_rate):
    x = layers.Dense(hidden_units[0])(x)
    x = layers.DepthwiseConv1D(3,data_format='channels_first',activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(hidden_units[1])(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

class SensorPatches(layers.Layer):
    def __init__(self, embedding_size, frame_size,timeStep, **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.frame_size = frame_size
        self.timeStep = timeStep
        self.embedding_size = embedding_size
        self.accEmbeds = layers.Conv1D(filters = int(embedding_size/2),kernel_size = frame_size,strides = timeStep, data_format = "channels_last")
        self.gyroEmbeds = layers.Conv1D(filters = int(embedding_size/2),kernel_size = frame_size,strides = timeStep, data_format = "channels_last")
    
    def call(self, inputData):
        accEmbedss = self.accEmbeds(inputData[:,:,:3])
        gyroEmbedss = self.gyroEmbeds(inputData[:,:,3:])
        concatEmbeds = tf.concat((accEmbedss,gyroEmbedss),axis=2)
        return concatEmbeds
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frame_size': self.frame_size,
            'embedding_size': self.embedding_size,
            'timeStep': self.timeStep,})
        return config

class PositionalEncoder(layers.Layer):
    def __init__(self,embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.embedding_size)

    def call(self, patches, training = None):
        
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1,1]
        )  # (B, num_patches, projection_dim)
        patch_embeddings = (patches + pos_embeddings)  # (B, num_patches, projection_dim)
        return patch_embeddings
    



def HART_student_encoder_LM(projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    # outouts = encoded_patches
    outouts = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, outouts, name="studentModel")    
    
    
def HART_student_encoder(projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        encoded_patches = layers.Add()([x3, x2])
    outouts = encoded_patches
    # outouts = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, outouts, name="studentModel")

    
def HART_teacher_encoder(projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1, layerAverage = 6):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    topKLayers = len(convKernels) - layerAverage
    topKEmbeddings = []
    
    inputs = layers.Input((None, projection_dim))
    encoded_patches = inputs
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(startIndex = projectionQuarter,
                          stopIndex = projectionQuarter + projectionHalf,
                          projectionSize = projectionHalf,
                          attentionHead =  filterAttentionHead, 
                          kernelSize = kernelLength,
                          dropPathRate = dropPathRate[layerIndex],
                          dropout_rate = dropout_rate,
                          name = "liteFormer_"+str(layerIndex))(x1)
        
        branch2Acc = SensorWiseMHA(projectionQuarter,num_heads,0,projectionQuarter,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate,name = "AccMHA_"+str(layerIndex))(x1)
        branch2Gyro = SensorWiseMHA(projectionQuarter,num_heads,projectionQuarter + projectionHalf ,projection_dim,dropPathRate = dropPathRate[layerIndex],dropout_rate = dropout_rate, name = "GyroMHA_"+str(layerIndex))(x1)
        concatAttention = tf.concat((branch2Acc,branch1,branch2Gyro),axis= 2 )
        x2 = layers.Add()([concatAttention, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp2(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)
        x3 = DropPath(dropPathRate[layerIndex])(x3)
        if(layerIndex >= topKLayers):
            topKEmbeddings.append(x3)
        encoded_patches = layers.Add()([x3, x2])
    stackedEmbedding = tf.stack(topKEmbeddings)
    averagegEmbedding = tf.math.reduce_mean(stackedEmbedding, axis=0)
    representation = layers.LayerNormalization(epsilon=1e-6)(averagegEmbedding)
    return tf.keras.Model(inputs, representation, name="teacherModel")

class EMA(tf.keras.callbacks.Callback):
    def __init__(self, decay= 0.9998):
        super(EMA, self).__init__()
        self.decay = decay
        self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)
    def on_train_begin(self, logs=None):
        self.ema.apply(self.model.studentEncoder.trainable_variables)
    def on_train_batch_end(self, batch, logs=None):
        train_vars = self.model.studentEncoder.trainable_variables
        averages = [self.ema.average(var) for var in train_vars]
        studentVariableLength = len(self.model.studentEncoder.trainable_variables)
        # for layers in self.model.teacherEncoder.layers:
        #     layers.trainable = True
        target_model_vars = self.model.teacherEncoder.trainable_variables[:studentVariableLength]
        assert len(target_model_vars) == len(averages)
        for i, var in enumerate(target_model_vars):
            var.assign(averages[i])
        # for layers in self.model.teacherEncoder.layers:
        #     layers.trainable = False
        self.ema.apply(self.model.studentEncoder.trainable_variables)



class FrameLayer(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super().__init__(**kwargs)
        self.frameLength = frameLength
        self.frameStride = frameStride

    def call(self, inputData, training=None):
        patchedData = tf.image.extract_patches(tf.expand_dims(inputData, 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        patchedData = layers.Reshape((-1,patchedData.shape[2]*patchedData.shape[3]))(patchedData)
        return patchedData
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frameLength': self.frameLength,
            'frameStride': self.frameStride,})
        return config
    
    
class PatchDemo(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super(PatchDemo, self).__init__(**kwargs)
        self.frameLength = frameLength
        self.frameStride = frameStride
    def call(self, inputData, training=None):
        patchedData = tf.image.extract_patches(tf.expand_dims(inputData, 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        patchedData = tf.transpose(patchedData, [0, 1, 3, 2])
        return patchedData
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frameLength': self.frameLength,
            'frameStride': self.frameStride,})
        return config

def ispl_inception_encoder(projection_dim,
                   filters_number = 64,
                   network_depth = 5,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                #    learning_rate = 0.01,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   metrics=['accuracy']):
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

    input_layer = layers.Input((None, projection_dim))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    m = tf.keras.Model(inputs=input_layer, outputs=x)

    return m


def ispl_inception_teacher_encoder(projection_dim,
                   filters_number = 64,
                   network_depth = 5,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   layerAverage = 2,
                   metrics=['accuracy']):
    weightinit = 'lecun_uniform'  # weight initialization

    topKLayers = network_depth - layerAverage
    topKEmbeddings = []
    
    def inception_module(input_tensor, stride=1, activation='relu'):

        # The  channel number is greater than 1
        if use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = layers.Conv1D(filters=bottleneck_size,
                                     kernel_size=1,
                                     padding='same',
                                     activation=activation,
                                     kernel_initializer=weightinit,

                                     use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_sizes = [max_kernel_size // (2 ** i) for i in range(3)]
        conv_list = []

        for kernel_size in kernel_sizes:
            conv_list.append(layers.Conv1D(filters=filters_number,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    activation=activation,
                                    kernel_initializer=weightinit,
                                    kernel_regularizer=l2(regularization_rate),
                                    use_bias=False)(input_inception))

        max_pool_1 = layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_last = layers.Conv1D(filters=filters_number,
                           kernel_size=1,
                           padding='same',
                           activation=activation,
                           kernel_initializer=weightinit,
                           kernel_regularizer=l2(regularization_rate),
                           use_bias=False)(max_pool_1)

        conv_list.append(conv_last)

        x = layers.Concatenate(axis=2)(conv_list)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def shortcut_layer(input_tensor, out_tensor):
        shortcut_y = layers.Conv1D(filters=int(out_tensor.shape[-1]),
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=weightinit,
                            kernel_regularizer=l2(regularization_rate),
                            use_bias=False)(input_tensor)
        shortcut_y = layers.BatchNormalization()(shortcut_y)

        x = layers.Add()([shortcut_y, out_tensor])
        x = layers.Activation('relu')(x)
        return x

    input_layer = layers.Input((None, projection_dim))
    # Build the actual model:
#     input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
            
        if(depth >= topKLayers):
            topKEmbeddings.append(x)
            
    stackedEmbedding = tf.stack(topKEmbeddings)
    averagegEmbedding = tf.math.reduce_mean(stackedEmbedding, axis=0)
    representation = layers.LayerNormalization(epsilon=1e-6)(averagegEmbedding)
    m = tf.keras.Model(inputs=input_layer, outputs=representation)

    return m


class MaskEncoder(layers.Layer):
    def __init__(self,embedding_size,mask_proportion,frame_length,channels = 6, **kwargs):
        super().__init__(**kwargs)
        self.mask_proportion = mask_proportion
        self.embedding_size = embedding_size
        self.mask_token = tf.Variable(
            tf.random.normal([1,frame_length * channels]), trainable=True
        )

    def build(self, input_shape):
        (_, self.num_patches, _) = input_shape
        self.num_mask = int(np.ceil(self.mask_proportion * self.num_patches))
        self.projection = layers.Dense(units=self.embedding_size)
        self.position_embedding = layers.Embedding(
            input_dim= self.num_patches, output_dim=self.embedding_size)
        print("Number of masked framed:")
        print(self.num_mask)
        
    def call(self, patches, dowmstreaming = True):
        
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1,1]
        ) 
        patch_embeddings = (self.projection(patches) + pos_embeddings)  
        if(dowmstreaming):
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)

            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)

            mask_tokens = tf.repeat(mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0)

            masked_positions = tf.gather(pos_embeddings, mask_indices, axis=1, batch_dims=1) 

            masked_token_embeddings = self.projection(mask_tokens) + masked_positions

            unmaskedEmbeds = tf.gather(patch_embeddings, unmask_indices, axis=1, batch_dims=1)  

            encoderInput = tf.concat((unmaskedEmbeds,masked_token_embeddings),axis = 1)

            return encoderInput,patch_embeddings,mask_indices
    
    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices


class SensorWiseMaskEncoder(layers.Layer):
    def __init__(self,embedding_size,mask_proportion,frame_length,channels = 6, downstream = False, **kwargs):
        super().__init__(**kwargs)
        self.mask_proportion = mask_proportion
        self.embedding_size = embedding_size
        self.mask_token = tf.Variable(
            tf.random.normal([1,frame_length * channels]), trainable=True
        )
        self.downstream = downstream
    def build(self, input_shape):
        (_, self.num_frames, self.frame_size ) = input_shape
        self.num_mask = int(np.ceil(self.mask_proportion * self.num_frames))
        self.accProjection = layers.Dense(units=self.embedding_size // 2)
        self.gyroProjection = layers.Dense(units=self.embedding_size // 2)
        self.position_embedding = layers.Embedding(
            input_dim= self.num_frames, output_dim=self.embedding_size)
        print("Number of masked framed:")
        print(self.num_mask)
        
    def call(self, framedInput, dowmstreaming = True):
        
        batch_size = tf.shape(framedInput)[0]
        positions = tf.range(start=0, limit=self.num_frames, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1,1]
        ) 
        projectedInput = tf.concat((self.accProjection(framedInput[:,:,:self.frame_size // 2]),self.gyroProjection(framedInput[:,:,self.frame_size // 2: ])),axis=2)
        patch_embeddings = (projectedInput + pos_embeddings)  
        if(dowmstreaming):
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            mask_tokens = tf.repeat(mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0)
            masked_positions = tf.gather(pos_embeddings, mask_indices, axis=1, batch_dims=1) 
            unmaskedEmbeds = tf.gather(patch_embeddings, unmask_indices, axis=1, batch_dims=1)  

            projected_mask_tokens = tf.concat((self.accProjection(mask_tokens[:,:,:self.frame_size // 2]),self.gyroProjection(mask_tokens[:,:,self.frame_size // 2: ])),axis=2)

            masked_token_embeddings = projected_mask_tokens + masked_positions
            
            encoderInput = tf.concat((unmaskedEmbeds,masked_token_embeddings),axis = 1)

            return encoderInput,patch_embeddings,mask_indices


    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_frames)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

class SensorWiseFrameLayer(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super(SensorWiseFrameLayer, self).__init__(**kwargs)
        self.frameLength = frameLength
        self.frameStride = frameStride

    def call(self, inputData, training=None):
        framedAccData = tf.image.extract_patches(tf.expand_dims(inputData[:,:,:3], 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        accEmbedss = tf.reshape(framedAccData, (-1,framedAccData.shape[1],framedAccData.shape[2]*framedAccData.shape[3]))
        framedGyroData = tf.image.extract_patches(tf.expand_dims(inputData[:,:,3:], 3), sizes=[1, self.frameLength, 1, 1], strides=[1, self.frameStride, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        gyroEmbeds = tf.reshape(framedGyroData, (-1,framedGyroData.shape[1],framedGyroData.shape[2]*framedGyroData.shape[3]))
        concatEmbeds = tf.concat((accEmbedss,gyroEmbeds),axis=2)
        return concatEmbeds
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'frameLength': self.frameLength,
            'frameStride': self.frameStride,})
        return config
    
class Data2Vec(tf.keras.Model):
    def __init__(
        self,
        sensorWiseEmbedder,
        maskEncoder,
        teacherEncoder,
        studentEncoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sensorWiseEmbedder = sensorWiseEmbedder
        self.maskEncoder = maskEncoder
        self.teacherEncoder = teacherEncoder
        self.studentEncoder = studentEncoder
        self.downstreamPooler = layers.GlobalAveragePooling1D()
        # self.layerNorm = layers.LayerNormalization()

    def return_feature_extrator(self):
        downStreamModel = tf.keras.Sequential(
        [
            layers.Input((128,6)),
            self.sensorWiseEmbedder,
            self.maskEncoder,
            self.studentEncoder,
            self.downstreamPooler,
        ])
        
        return downStreamModel
    
    def call(self,inputData):
        patchedData = self.sensorWiseEmbedder(inputData)
        embededData = self.maskEncoder(patchedData)
        encoderEmbeddings = self.studentEncoder(embededData)
        studentOutputs = self.downstreamPooler(encoderEmbeddings)
        return studentOutputs
    
    
    def calculate_loss(self, inputData, dowmstreaming = False):
        patchedData = self.sensorWiseEmbedder(inputData)
        studentInput,patch_embeddings,mask_indices = self.maskEncoder(patchedData,dowmstreaming = dowmstreaming)
        teacherOutputs = tf.gather(tf.stop_gradient(self.teacherEncoder(patch_embeddings)),mask_indices,axis=1, batch_dims=1)        
        mainStartIndex = studentInput.shape[1] - mask_indices.shape[1]
        studentOutputs = self.studentEncoder(studentInput)[:,mainStartIndex:,:]        
        total_loss = self.compiled_loss(teacherOutputs, studentOutputs)
        return total_loss, teacherOutputs, studentOutputs
    
    def train_step(self, inputData):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(inputData)

        # Apply gradients.
        train_vars = [
            self.sensorWiseEmbedder.trainable_variables,
            self.maskEncoder.trainable_variables,
            self.studentEncoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, inputData):
        total_loss, loss_patch, loss_output = self.calculate_loss(inputData[0])
        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}
