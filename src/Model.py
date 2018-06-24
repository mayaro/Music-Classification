from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization

def AudioModel( input_shape ):
  kernel_size = 3
  
  activation_function = Activation('relu')
  inputs = Input(input_shape)

  conv_layer_1 = Conv1D(128, kernel_size)(inputs)
  activation_layer_1 = activation_function(conv_layer_1)
  normalization_layer_1 = BatchNormalization( )(activation_layer_1)
  pooling_layer_1 = AveragePooling1D(pool_size = 4, strides = 2)(normalization_layer_1)

  conv_layer_2 = Conv1D(256, kernel_size)(pooling_layer_1)
  activation_layer_2 = activation_function(conv_layer_2)
  normalization_layer_2 = BatchNormalization( )(activation_layer_2)
  pooling_layer_2 = AveragePooling1D(pool_size = 2, strides = 2)(normalization_layer_2)

  conv_layer_3 = Conv1D(256, kernel_size)(pooling_layer_2)
  activation_layer_3 = activation_function(conv_layer_3)
  bn3 = BatchNormalization( )(activation_layer_3)
  pooling_layer_3 = AveragePooling1D( )(bn3)

  conv_layer_4 = Conv1D(512, kernel_size)(pooling_layer_3)
  activation_layer_4 = activation_function(conv_layer_4)
  normalization_layer_4 = BatchNormalization( )(activation_layer_4)

  global_max_pooling_layer = GlobalMaxPooling1D( )(normalization_layer_4)
  global_mean_pooling_layer = GlobalAveragePooling1D( )(normalization_layer_4)
  mergedlayer = concatenate([ global_max_pooling_layer, global_mean_pooling_layer ], axis = 1)

  fully_connected_layer = Dense(512,
                  kernel_initializer = 'glorot_normal',
                  bias_initializer = 'glorot_normal')(mergedlayer)
  fc_activation_layer = activation_function(fully_connected_layer)
  regulation_dropout_layer = Dropout(0.75)(fc_activation_layer)

  classification_layer = Dense(5, activation = 'softmax')(regulation_dropout_layer)

  model = Model(inputs = [ inputs ], outputs = [ classification_layer ])
  return model