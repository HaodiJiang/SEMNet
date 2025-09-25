# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   Sam Houston State University
#   Huntsville, Texas 77341, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU

from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.activations import swish

def resnet_block(input_tensor, filters, strides=1, downsample=False):
    """
    A basic residual block for ResNet.
    Args:
        input_tensor: Input tensor to the block.
        filters: Number of filters for the convolutional layers.
        strides: Stride for the first convolutional layer.
        downsample: If True, downsample the input tensor with a Conv2D layer.
    """
    shortcut = input_tensor

    # First convolution layer
    x = Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # x = Activation(swish)(x)

    # Second convolution layer
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # If downsampling is required, adjust the shortcut (identity connection)
    if downsample:
        shortcut = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Add shortcut to the output of the second convolution
    x = Add()([x, shortcut])
    # x = Activation(swish)(x)
    x = LeakyReLU(alpha=0.2)(x)

    return x

# version 1
class SEMNet(object):

    def __init__(self, image_shape):
        self.image_shape = image_shape

    def cnn_model(self):
        input = Input(shape=self.image_shape)

        # Initial Conv Layer
        x = Conv2D(64, kernel_size=7, strides=2, padding='same')(input)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

        # First block of residual layers (64 filters)
        x = resnet_block(x, filters=64)
        x = resnet_block(x, filters=64)

        # Second block of residual layers (128 filters, with downsampling)
        x = resnet_block(x, filters=128, strides=2, downsample=True)
        x = resnet_block(x, filters=128)
        x = resnet_block(x, filters=128)

        # Third block of residual layers (256 filters, with downsampling)
        x = resnet_block(x, filters=256, strides=2, downsample=True)
        x = resnet_block(x, filters=256)
        x = resnet_block(x, filters=256)

        # Fourth block of residual layers (512 filters, with downsampling)
        x = resnet_block(x, filters=512, strides=2, downsample=True)
        x = resnet_block(x, filters=512)

        # Global Average Pooling
        x = GlobalAveragePooling2D()(x)

        # Add dropout for regularization
        # Setting training=True ensures dropout is applied even during inference.
        x = Dropout(0.2)(x, training=True)

        # Fully connected layer with 2 outputs
        x = Dense(2)(x)
        output = Activation('linear')(x)

        # Create the model
        model = Model(inputs=input, outputs=output)

        return model



# # Instantiate the model and print the summary
# image_shape = (256, 256, 3)  # Example input shape for ResNet
# model_instance = myResNet_with_dropout_trainable_Model(image_shape)
# model = model_instance.cnn_model()
# model.summary()