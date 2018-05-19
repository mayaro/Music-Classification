from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization

class TrainingModels(object):

  @staticmethod
  def CNNMelspec( input_shape ):
    kernel_size = 3
    
    activation_function = Activation('relu')
    inputs = Input(input_shape)

    conv_block_1 = Conv1D(32, kernel_size)(inputs)
    activation_1 = activation_function(conv_block_1)
    normalization_1 = BatchNormalization( )(activation_1)
    pool1 = AveragePooling1D(pool_size = 2, strides = 2)(normalization_1)

    conv2 = Conv1D(64, kernel_size)(pool1)
    act2 = activation_function(conv2)
    bn2 = BatchNormalization( )(act2)
    pool2 = AveragePooling1D(pool_size = 2, strides = 2)(bn2)

    conv3 = Conv1D(128, kernel_size)(pool2)
    act3 = activation_function(conv3)
    bn3 = BatchNormalization( )(act3)

    gmaxpl = GlobalMaxPooling1D( )(bn3)
    gmeanpl = GlobalAveragePooling1D( )(bn3)
    mergedlayer = concatenate([ gmaxpl, gmeanpl ], axis = 1)

    fully_connected_1 = Dense(512,
                   kernel_initializer = 'glorot_normal',
                   bias_initializer = 'glorot_normal')(mergedlayer)
    mlp_activation_function = activation_function(fully_connected_1)
    reg = Dropout(0.5)(mlp_activation_function)

    fully_connected_2 = Dense(512,
                   kernel_initializer = 'glorot_normal',
                   bias_initializer = 'glorot_normal')(reg)
    mlp_activation_function = activation_function(fully_connected_2)
    reg = Dropout(0.5)(mlp_activation_function)

    fully_connected_2 = Dense(5, activation = 'softmax')(reg)

    model = Model(inputs = [ inputs ], outputs = [ fully_connected_2 ])
    return model