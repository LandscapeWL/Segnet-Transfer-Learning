import keras
from keras.models import *
from keras.layers import *
from keras import layers
import keras.backend as K

IMAGE_ORDERING = 'channels_last'
def one_side_pad( x ):
    x = ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING)(x)
    if IMAGE_ORDERING == 'channels_first':
        x = Lambda(lambda x : x[: , : , :-1 , :-1 ] )(x)
    elif IMAGE_ORDERING == 'channels_last':
        x = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters
    
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 1x1 compression
    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING , name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3 extracted features
    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING ,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 1 x 1 expansion characteristics
    x = Conv2D(filters3 , (1, 1), data_format=IMAGE_ORDERING , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # Residual networks
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# The biggest difference with identity_block is that it can reduce wh and compress
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    #64,64,256
    filters1, filters2, filters3 = filters
    
    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 1x1 compression, dimensionality reduction
    x = Conv2D(filters1, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3 extracted features
    x = Conv2D(filters2, kernel_size , data_format=IMAGE_ORDERING  , padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 1x1 expansion of features, ascension
    x = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    #Compressed input image, residual edge
    # 1 x 1 expansion characteristics
    shortcut = Conv2D(filters3, (1, 1) , data_format=IMAGE_ORDERING  , strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    # add
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def get_resnet50_encoder(input_height=224 ,  input_width=224 , pretrained='imagenet' , 
                                include_top=True, weights='imagenet',
                                input_tensor=None, input_shape=None,
                                pooling=None,
                                classes=1000):

    assert input_height%32 == 0
    assert input_width%32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3,input_height,input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height,input_width , 3 ))


    if IMAGE_ORDERING == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1


    x = ZeroPadding2D((3, 3), data_format=IMAGE_ORDERING)(img_input)
    #224,224,3  ->  112,112,64
    x = Conv2D(64, (7, 7), data_format=IMAGE_ORDERING, strides=(2, 2), name='conv1')(x)
    # f1 is the result of one compression in the hw direction
    f1 = x

    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # 112,112,64  ->  56,56,64
    x = MaxPooling2D((3, 3) , data_format=IMAGE_ORDERING , strides=(2, 2))(x)
    
    # 56,56,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # f2 is the result of two compressions in the hw direction
    f2 = one_side_pad(x )

    # 28,28,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # 3 is the result of three compressions in the hw direction
    f3 = x 

    # 14,14,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # f4 is the result of four compressions in the hw direction
    f4 = x 

    # 7,7,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # f5 is the result of five compressions in the hw direction
    f5 = x 

    # Replaces the fully connected layer
    x = AveragePooling2D((7, 7) , data_format=IMAGE_ORDERING , name='avg_pool')(x)

    return img_input , [f1 , f2 , f3 , f4 , f5  ]
