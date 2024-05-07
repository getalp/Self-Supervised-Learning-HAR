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
    
class MaskEncoder(layers.Layer):
    def __init__(self,embedding_size,mask_proportion, **kwargs):
        super().__init__(**kwargs)
        self.mask_proportion = mask_proportion
        self.mask_token = tf.Variable(
            tf.random.normal([1,embedding_size]), trainable=True
        )
    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
        self.num_mask = int(np.ceil(self.mask_proportion * self.num_patches))
        print("Number of mask")
        print(self.num_mask)
    def call(self, patches, training = None):
        batch_size = tf.shape(patches)[0]
        mask_indices, unmask_indices = self.get_random_indices(batch_size)
        mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
        mask_tokens = tf.repeat(mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0)
        unmaskedEmbeds = tf.gather(patches, unmask_indices, axis=1, batch_dims=1)  
        encoderInput = tf.concat((unmaskedEmbeds,mask_tokens),axis = 1)
        return encoderInput,mask_indices
    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

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
    # Build the actual model:
#     input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    m = tf.keras.Model(inputs=input_layer, outputs=x)

    return m

def ispl_inception_decoder(
                   enc_embedding_size,
                   patch_count, 
                   output_shape,
                   filters_number = 64,
                   network_depth = 3,
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
    
    inputs = layers.Input((patch_count, enc_embedding_size))

    x = layers.BatchNormalization()(inputs)  # Added batchnorm (not in original paper)
    input_res = x

    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = layers.GlobalAveragePooling1D()(x)
    
    
#     representation = layers.Flatten()(representation)
    pre_final = layers.Dense(units=output_shape[0] * output_shape[1])(gap_layer)
    outputs = layers.Reshape(output_shape)(pre_final)
    return tf.keras.Model(inputs, outputs, name="mae_decoder")



class PatchLayer(layers.Layer):
    def __init__(self, frameLength, frameStride,**kwargs):
        super(PatchLayer, self).__init__(**kwargs)
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

def HART_decoder(enc_embedding_size,patch_count = 8, output_shape = (128,6), frame_length = 16, projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 7, 15],dropout_rate = 0.1,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input((patch_count, enc_embedding_size))
    encoded_patches = layers.Dense(projection_dim)(inputs)
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
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    pre_final = layers.Dense(units=output_shape[0] * output_shape[1])(representation)
    outputs = layers.Reshape(output_shape)(pre_final)
#     outputs = PatchLayer(frame_length,frame_length)(pre_final)
    return tf.keras.Model(inputs, outputs, name="mae_decoder")    
    
def HART_encoder(projection_dim = 192,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31],dropout_rate = 0.1,useTokens = False):
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
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, representation, name="mae_encoder")    
    
class SensorWisePatchEncoder(layers.Layer):
    def __init__(self,patch_size,embedding_size,mask_proportion,channels = 6, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.mask_proportion = mask_proportion
        self.mask_token = tf.Variable(
            tf.random.normal([1, channels * patch_size]), trainable=True
        )
    def build(self, input_shape):
        (_, self.num_frames, self.frame_size) = input_shape
        self.accProjection = layers.Dense(units=self.embedding_size // 2)
        self.gyroProjection = layers.Dense(units=self.embedding_size // 2)
        
        self.num_mask = int(self.mask_proportion * self.num_frames)
        self.position_embedding = layers.Embedding(
            input_dim=self.num_frames, output_dim=self.embedding_size)

    def call(self, framedInput, downstream = True, training = None):
        
        batch_size = tf.shape(framedInput)[0]
        positions = tf.range(start=0, limit=self.num_frames, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1,1]
        )  
        projectedInput = tf.concat((self.accProjection(framedInput[:,:,:self.frame_size // 2]),self.gyroProjection(framedInput[:,:,self.frame_size // 2: ])),axis=2)
        patch_embeddings = (projectedInput + pos_embeddings)  # (B, num_frames, projection_dim)
#         training=True
        if(downstream):
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            unmasked_embeddings = tf.gather(patch_embeddings, unmask_indices, axis=1, batch_dims=1)  

            unmasked_positions = tf.gather(pos_embeddings, unmask_indices, axis=1, batch_dims=1)  # (B, unmask_numbers, projection_dim)

            masked_positions = tf.gather(pos_embeddings, mask_indices, axis=1, batch_dims=1)  # (B, mask_numbers, projection_dim)

            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            
            mask_tokens = tf.repeat(
                    mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
                )
            mask_tokens = tf.concat((self.accProjection(mask_tokens[:,:,:self.frame_size // 2]),self.gyroProjection(mask_tokens[:,:,self.frame_size // 2: ])),axis=2)
            masked_embeddings =  mask_tokens + masked_positions
            
            return (
                    unmasked_embeddings,  # Input to the encoder.
                    masked_embeddings,  # First part of input to the decoder.
                    unmasked_positions,  # Added to the encoder outputs.
                    mask_indices,  # The indices that were masked.
                    unmask_indices,  # The indices that were unmaksed.
                )
    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_frames)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patch, unmask_indice):
        new_patch = np.zeros_like(patch)
        count = 0
        for i in range(unmask_indice.shape[0]):
            new_patch[unmask_indice[i]] = patch[unmask_indice[i]]
        return new_patch    
    
class PatchEncoder(layers.Layer):
    def __init__(self,patch_size,embedding_size,mask_proportion,channels = 6, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self.patch_size = patch_size
        self.mask_proportion = mask_proportion
        self.mask_token = tf.Variable(
            tf.random.normal([1, channels * patch_size]), trainable=True
        )
#         self.downstream = downstream 
    
    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape
        self.projection = layers.Dense(units=self.embedding_size)
        self.num_mask = int(self.mask_proportion * self.num_patches)

        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.embedding_size)

    def call(self, patches, downstream = True, training = None):
        
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1,1]
        )  # (B, num_patches, projection_dim)
        
        
        patch_embeddings = self.projection(patches)
        patch_embeddings = (patch_embeddings + pos_embeddings)  # (B, num_patches, projection_dim)
#         training=True
        if(downstream):
            return patch_embeddings
        else:
            mask_indices, unmask_indices = self.get_random_indices(batch_size)
            unmasked_embeddings = tf.gather(patch_embeddings, unmask_indices, axis=1, batch_dims=1)  

            unmasked_positions = tf.gather(pos_embeddings, unmask_indices, axis=1, batch_dims=1)  # (B, unmask_numbers, projection_dim)

            masked_positions = tf.gather(pos_embeddings, mask_indices, axis=1, batch_dims=1)  # (B, mask_numbers, projection_dim)

            mask_tokens = tf.repeat(self.mask_token, repeats=self.num_mask, axis=0)
            
            mask_tokens = tf.repeat(
                    mask_tokens[tf.newaxis, ...], repeats=batch_size, axis=0
                )
            mask_tokens = self.projection(mask_tokens)
            masked_embeddings =  mask_tokens + masked_positions
            
            return (
                    unmasked_embeddings,  # Input to the encoder.
                    masked_embeddings,  # First part of input to the decoder.
                    unmasked_positions,  # Added to the encoder outputs.
                    mask_indices,  # The indices that were masked.
                    unmask_indices,  # The indices that were unmaksed.
                )
    def get_random_indices(self, batch_size):
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        mask_indices = rand_indices[:, : self.num_mask]
        unmask_indices = rand_indices[:, self.num_mask :]
        return mask_indices, unmask_indices

    def generate_masked_image(self, patch, unmask_indice):
        new_patch = np.zeros_like(patch)
        count = 0
        for i in range(unmask_indice.shape[0]):
            new_patch[unmask_indice[i]] = patch[unmask_indice[i]]
        return new_patch
    
    
    
class MaskedAutoencoder(tf.keras.Model):
    def __init__(
        self,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.downstreamPooler = layers.GlobalAveragePooling1D()
    def return_feature_extrator(self):
        downStreamModel = tf.keras.Sequential(
        [
            layers.Input((128,6)),
            self.patch_layer,
            self.patch_encoder,
            self.encoder,
            self.downstreamPooler,
        ])
        return downStreamModel
    
    def call(self,inputData):
        patches = self.patch_layer(inputData)
        patch_embeddings = self.patch_encoder(patches)
        encoder_outputs = self.encoder(patch_embeddings)
        embeddings_output = self.downstreamPooler(encoder_outputs)
        return embeddings_output
    
    def calculate_loss(self, inputData, training = False, downstream = True):
        patches = self.patch_layer(inputData)
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches, downstreaming = downstream, training = training)

        # Pass the unmaksed patche to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings, training = training)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_patches = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_patches)

#         decoder_patches = self.patch_layer(decoder_outputs)
        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Compute the total loss.
        total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output
        
    def train_step(self, inputData):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(inputData,training = True, downstream = False)

        # Apply gradients.
        train_vars = [
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
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
    def test_step(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images[0], training = False,downstream = False)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}
