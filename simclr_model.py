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
        inputShape = tf.shape(formattedInputs)
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
#     outouts = encoded_patches
    outouts = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    return tf.keras.Model(inputs, outouts, name="studentModel")

    
def ispl_inception_encoder(x_shape,
                   filters_number = 64,
                   network_depth = 5,
                   use_residual = True,
                   use_bottleneck = True,
                   max_kernel_size = 68,
                #    learning_rate = 0.01,
                   bottleneck_size = 32,
                   regularization_rate = 0.00593,
                   metrics=['accuracy']):
    dim_length = x_shape[0]  # number of samples in a time series
    dim_channels = x_shape[1]  # number of channels
    weightinit = 'lecun_uniform'  # weight initialization

    def inception_module(input_tensor, stride=1, activation='relu'):

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
    # Build the actual model:
    input_layer = layers.Input((dim_length, dim_channels))
    x = layers.BatchNormalization()(input_layer)  # Added batchnorm (not in original paper)
    input_res = x
    for depth in range(network_depth):
        x = inception_module(x)

        if use_residual and depth % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
    gap_layer = layers.GlobalAveragePooling1D()(x)
    m = tf.keras.Model(inputs=input_layer, outputs=gap_layer)
    return m


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
    
    



class SimCLR(tf.keras.Model):
    def __init__(
        self,
        encoder,
        projection_heads,
        transformation_function,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.projection_heads = projection_heads
        self.transformation_function = transformation_function
        
    def return_feature_extrator(self):
        downStreamModel = tf.keras.Sequential(
        [
            layers.Input((128,6)),
            self.encoder
        ])
        return downStreamModel
    
    def call(self,inputData):
        # transfromedInput1 = self.transformation_function(inputData)
        # transfromedInput2 = self.transformation_function(inputData)
        # total_loss,loss_patch, loss_output = self.calculate_loss(transfromedInput1,transfromedInput2)
        output = self.encoder(inputData)
        return output

    def calculate_loss(self, transfromedInput1,transfromedInput2, normalize=True, temperature=1.0, weights=1.0):
        embed1 = self.encoder(transfromedInput1)
        projection1 = self.projection_heads(embed1)
        embed2 = self.encoder(transfromedInput2)
        projection2 = self.projection_heads(embed2)
        loss = self.compiled_loss(projection1, projection2)
        return loss,projection1,projection2
        
    def train_step(self, inputData):
        transfromedInput1 = self.transformation_function(inputData)
        transfromedInput2 = self.transformation_function(inputData)
        with tf.GradientTape() as tape:
            total_loss,loss_patch, loss_output = self.calculate_loss(transfromedInput1,transfromedInput2)

        # Apply gradients.
        train_vars = [
            self.encoder.trainable_variables,
            self.projection_heads.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, inputTuple):
        transfromedInput1 = self.transformation_function(inputTuple[0])
        transfromedInput2 = self.transformation_function(inputTuple[0])
        total_loss, loss_patch, loss_output = self.calculate_loss(transfromedInput1,transfromedInput2)
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}
    
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim,**kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = patch + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,})
        return config
    
class SensorPatches(layers.Layer):
    def __init__(self, projection_dim, patchSize,timeStep, **kwargs):
        super(SensorPatches, self).__init__(**kwargs)
        self.patchSize = patchSize
        self.timeStep = timeStep
        self.projection_dim = projection_dim
        self.accProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
        self.gyroProjection = layers.Conv1D(filters = int(projection_dim/2),kernel_size = patchSize,strides = timeStep, data_format = "channels_last")
    def call(self, inputData):

        accProjections = self.accProjection(inputData[:,:,:3])
        gyroProjections = self.gyroProjection(inputData[:,:,3:])
        Projections = tf.concat((accProjections,gyroProjections),axis=2)
        return Projections
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patchSize': self.patchSize,
            'projection_dim': self.projection_dim,
            'timeStep': self.timeStep,})
        return config
    
def HART_encoder(input_shape, projection_dim = 192,patchSize = 16,timeStep = 16,num_heads = 3,filterAttentionHead = 4, convKernels = [3, 7, 15, 31, 31, 31], mlp_head_units = [1024],dropout_rate = 0.3,useTokens = False):
    projectionHalf = projection_dim//2
    projectionQuarter = projection_dim//4
    dropPathRate = np.linspace(0, dropout_rate* 10, len(convKernels)) * 0.1
    transformer_units = [
    projection_dim * 2,
    projection_dim,]  
    inputs = layers.Input(shape=input_shape)
    patches = SensorPatches(projection_dim,patchSize,timeStep)(inputs)
    patchCount = patches.shape[1] 
    encoded_patches = PatchEncoder(patchCount, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for layerIndex, kernelLength in enumerate(convKernels):        
        x1 = layers.LayerNormalization(epsilon=1e-6 , name = "normalizedInputs_"+str(layerIndex))(encoded_patches)
        branch1 = liteFormer(
                          startIndex = projectionQuarter,
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
    averagedOutputs = layers.GlobalAveragePooling1D()(representation)
    model = tf.keras.Model(inputs=inputs, outputs=averagedOutputs)
    return model

def projection_head(enc_embedding_size,hidden_1=256, hidden_2=128, hidden_3=50):
    inputs = layers.Input((enc_embedding_size))
    projection_1 = tf.keras.layers.Dense(hidden_1, activation=tf.nn.swish)(inputs)
    projection_2 = tf.keras.layers.Dense(hidden_2, activation=tf.nn.swish)(projection_1)
    outputs = tf.keras.layers.Dense(hidden_3,activation=tf.nn.swish)(projection_2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def scaling_transform_vectorized(X, sigma=0.1):
    """
    Scaling by a random factor
    """
    scaling_factor = tf.random.normal(shape=(tf.shape(X)[0], 1, tf.shape(X)[2]), mean=1.0, stddev=sigma, dtype=tf.float32)
    return X * scaling_factor

def noise_transform_vectorized(X, sigma=0.05):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = tf.random.normal(shape=tf.shape(X), mean=0, stddev=sigma, dtype=tf.float32)
    return X + noise
def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    inputShape = tf.shape(X)
    axes = tf.random.uniform((inputShape[0], 3),minval= -1,maxval= 1)
    angles = tf.random.uniform([inputShape[0]],minval=np.pi,maxval=np.pi)
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)
    temp1 = tf.linalg.matmul(X[:,:,:3], matrices)
    temp2 = tf.linalg.matmul(X[:,:,3:], matrices)
    concatenateTemp = tf.concat([temp1,temp2],axis = 2)
    return concatenateTemp

def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes

    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / tf.norm(axes, ord=2, axis=1, keepdims=True)
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = tf.math.cos(angles)
    s = tf.math.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = tf.convert_to_tensor([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = tf.transpose(m, perm=[2,0,1])
    return matrix_transposed
    
class NT_Xent_loss(tf.keras.losses.Loss):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.LARGE_NUM = 1e9

    def call(self, hidden_features_transform_1, hidden_features_transform_2):        
        entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        batch_size = tf.shape(hidden_features_transform_1)[0]

        h1 = tf.math.l2_normalize(hidden_features_transform_1, axis=1)
        h2 = tf.math.l2_normalize(hidden_features_transform_2, axis=1)

        labels = tf.range(batch_size)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(h1, h1, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = tf.matmul(h2, h2, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = tf.matmul(h1, h2, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(h2, h1, transpose_b=True) / self.temperature

        loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b
        
        return loss

def generate_composite_transform_function_simple(transform_funcs):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions
            the function is composed by applying 
            transform_funcs[0] -> transform_funcs[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """

    def combined_transform_func(sample):
        for func in transform_funcs:
            sample = func(sample)
        return sample
    return combined_transform_func