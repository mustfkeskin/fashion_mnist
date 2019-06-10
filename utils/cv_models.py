import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers import CuDNNLSTM as LSTM # 3 times faster than traditional LSTM
from keras.layers import Input, add, Add
from keras.layers.merge import concatenate


# https://keras.io/applications/
from keras.applications.mobilenet_v2 import MobileNetV2 #https://arxiv.org/abs/1801.04381
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50



def buildMLP(num_classes = 10):

	model = Sequential()
	model.add(Dense(512, input_shape=(784,), activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	return model


def buildSimpleCNN(num_classes=10, input_shape = (28,28,1)):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					activation='relu',
					kernel_initializer='he_normal',
					input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(64, 
					kernel_size=(3, 3), 
					activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))


	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer='adam',
				metrics=['accuracy'])

	return model



def buildCNNDropout(num_classes=10, input_shape = (28,28,1)):

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					activation='relu',
					kernel_initializer='he_normal',
					input_shape=input_shape))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Dropout(0.4))
	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))

	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])
	return model


def buildCNNBatchNorm(num_classes=10, input_shape = (28,28,1)):

	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
					activation='relu', 
					kernel_initializer='he_normal',
					input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))

	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.Adam(),
				metrics=['accuracy'])

	return model


def buildCNN(num_classes=10, input_shape = (32,32,1)):

	model = Sequential()

	# CONV => RELU => CONV => RELU => POOL layer set	
	model.add(Conv2D(32, (3, 3), padding="same",
		input_shape=input_shape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(32, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# second CONV => RELU => CONV => RELU => POOL layer set
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	# softmax classifier
	model.add(Dense(num_classes), Activation("softmax"))
	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	# return the constructed network architecture
	return model


def buildMobileNetV2(num_classes=10, input_shape = (32,32,1)):
	mobileNetV2 = MobileNetV2(input_shape=input_shape,
                          alpha=1.0,
                          depth_multiplier=1,
                          include_top=True,
                          weights=None,
                          input_tensor=None,
                          pooling=None,
                          classes=num_classes)

	mobileNetV2.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	return mobileNetV2


def buildNASNet(num_classes=10, input_shape = (32,32,1)):

	nasNet = NASNetMobile(input_shape=input_shape,
                      include_top=True,
                      weights=None,
                      input_tensor=None,
                      pooling=None,
                      classes=num_classes)

	nasNet.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	return nasNet


def buildResNet50(num_classes=10, input_shape = (32,32,1)):

	resNet50 = ResNet50(include_top=True,
							 weights=None,
							 input_tensor=None,
							 input_shape=input_shape,
							 pooling=None,
							 classes=num_classes)

	resNet50.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])
	return resNet50

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):

	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)

	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	layer_in = BatchNormalization()(layer_in)
	layer_in = Dropout(0.25)(layer_in)
	return layer_in

def buildSimpleVGG(num_classes=10, input_shape = (32,32,1)):

	visible = Input(shape=input_shape)

	layer = vgg_block(visible, 32, 2)
	layer = vgg_block(layer, 64, 2)
	layer = vgg_block(layer, 128, 2)

	layer = Flatten()(layer)
	layer = Dense(128, activation='relu')(layer)
	layer = Dropout(0.3)(layer)

	output = Dense(num_classes, activation='softmax')(layer)

	model = Model(inputs=visible, outputs=output)
	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	return model

# function for creating a naive inception block
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

def buildSimpleInception(num_classes=10, input_shape = (32,32,1)):

	# define model input
	visible = Input(shape=input_shape)
	
	# add inception block 1
	layer = inception_module(visible, 16, 24, 32, 4, 8, 8)
	
	# BN
	layer = BatchNormalization()(layer)
	layer = Dropout(0.3)(layer)
	
	# add inception block 2
	layer = inception_module(layer, 32, 32, 48, 8, 24, 16)
	
	# create model
	layer = Flatten()(layer)
	layer = Dense(64, activation='relu')(layer)
	layer = Dropout(0.3)(layer)
	
	output = Dense(num_classes, activation='softmax')(layer)

	model = Model(inputs=visible, outputs=output)
	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])
	
	return model


def residual_module(layer_in, n_filters):

	merge_input = layer_in
	
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	
	# conv1
	conv1 = Conv2D(n_filters, (3,3), dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	
	# conv2
	conv2 = Conv2D(n_filters, (3,3), dilation_rate=2, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)

	conv2 = BatchNormalization()(conv2)
	conv2 = Dropout(0.3)(conv2)

	# add filters, assumes filters/channels last
	layer_out = add([conv2, merge_input])

	# activation function
	layer_out = Activation('relu')(layer_out)
	return layer_out

def buildSimpleResnet(num_classes=10, input_shape = (32,32,1)):

	# define model input
	visible = Input(shape=input_shape)

	# add vgg module
	layer = residual_module(visible, 64)

	# create model
	layer = Flatten()(layer)
	layer = Dense(64, activation='relu')(layer)
	layer = Dropout(0.3)(layer)

	output = Dense(num_classes, activation='softmax')(layer)

	model = Model(inputs=visible, outputs=output)
	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])
	return model




def wide_residual_block(x, filters, n, strides, dropout):

	# Normal part
	x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)# , kernel_regularizer=l2(5e-4)
	x_res = BatchNormalization()(x_res)
	x_res = Activation('relu')(x_res)
	x_res = Conv2D(filters, (3,3), padding="same")(x_res)

	# Alternative branch
	x = Conv2D(filters, (1,1), strides=strides)(x)

	# Merge Branches
	x = Add()([x_res, x])

	for i in range(n-1):
		# Residual conection
		x_res = BatchNormalization()(x)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)

		# Apply dropout if given
		if dropout: x_res = Dropout(dropout)(x)

		# Second part
		x_res = BatchNormalization()(x_res)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)

		# Merge branches
		x = Add()([x, x_res])

	# Inter block part
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def build_wide_resnet(n, k, num_classes=10, input_shape = (32,32,1),  act= "relu", dropout=None):
	""" Builds the model. Params:
			- n: number of layers. WRNs are of the form WRN-N-K
				 It must satisfy that (N-4)%6 = 0
			- k: Widening factor. WRNs are of the form WRN-N-K
				 It must satisfy that K%2 = 0
			- input_dims: input dimensions for the model
			- output_dim: output dimensions for the model
			- dropout: dropout rate - default=0 (not recomended >0.3)
			- act: activation function - default=relu. Build your custom
				   one with keras.backend (ex: swish, e-swish)
	"""
	# Ensure n & k are correct
	assert (n-4)%6 == 0
	assert k%2 == 0
	n = (n-4)//6 
	# This returns a tensor input to the model
	inputs = Input(shape=input_shape)

	# Head of the model
	x = Conv2D(16, (3,3), padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# 3 Blocks (normal-residual)
	x = wide_residual_block(x, 16*k, n, (1,1), dropout) # 0
	x = wide_residual_block(x, 32*k, n, (2,2), dropout) # 1
	x = wide_residual_block(x, 64*k, n, (2,2), dropout) # 2
			
	# Final part of the model
	x = AveragePooling2D((8,8))(x)
	x = Flatten()(x)
	outputs = Dense(num_classes, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=outputs)
	model.compile(loss=keras.losses.categorical_crossentropy,
		          optimizer=keras.optimizers.Adam(),
		          metrics=['accuracy'])

	return model


def buildStacking(num_classes=10):

	model = Sequential()

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=  ['accuracy'])
	return model


def buildCNNLstm(num_classes=10, input_shape = (28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, kernel_size=(3, 3), padding="same",
					activation='relu', 
					kernel_initializer='he_normal',
					input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(Flatten())

	modelLstm = Sequential()
	#modelLstm.add(TimeDistributed(model, ...))
	model.add(LSTM(128, return_sequences = True))
	model.add(Dropout(0.3))

	model.add(Dense(num_classes, activation='softmax'))
	return model