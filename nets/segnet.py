from keras.models import *
from keras.layers import *
from nets.resnet50 import get_resnet50_encoder
IMAGE_ORDERING = 'channels_last'
def segnet_decoder(  f , n_classes , n_up=3 ):

	assert n_up >= 2

	o = f
	o = ( ZeroPadding2D( (1,1) , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	# Perform an UpSampling2D, where hw becomes 1/8 of its original size
	o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	# Perform an UpSampling2D, where hw becomes 1/4 of its original size
	for _ in range(n_up-2):
		o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ) )(o)
		o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
		o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ))(o)
		o = ( BatchNormalization())(o)

	# Perform an UpSampling2D, where hw becomes 1/4 of its original size
	o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ))(o)
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)

	# The output is h_input/2,w_input/2,nclasses
	o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o )
	
	return o 

def _segnet( n_classes , encoder  ,  input_height=416, input_width=608 , encoder_level=3):
	# encoder via backbone network
	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )

	# Get the result of hw compression four times
	feat = levels[encoder_level]

	# Passing features into the segnet network
	o = segnet_decoder(feat, n_classes, n_up=3 )

	# Reshape the result
	o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)
	o = Softmax()(o)
	model = Model(img_input,o)

	return model

def resnet50_segnet( n_classes ,  input_height=416, input_width=416 , encoder_level=3):

	model = _segnet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width , encoder_level=encoder_level)
	model.model_name = "resnet50_segnet"
	return model

