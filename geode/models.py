# models.py

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D



class Unet(tf.keras.Model):

    def __init__(self, n_channels: int = 3,
                 n_classes: int = 2,
                 n_filters: int = 64,
                 dropout_rate: float = 0.2):

        # define attributes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        # initialize the Model superclass
        super().__init__()

        # define the different Unet layers
        self.input_layer = Input(shape=(None, None, self.n_classes),
                                 dtype=tf.float32)

        self.conv_0 = Conv2D(filters=self.n_filters,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu')

        self.conv_1 = Conv2D(filters=2 * self.n_filters,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu')

        self.conv_2 = Conv2D(filters=4 * self.n_filters,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu')

        self.conv_3 = Conv2D(filters=8 * self.n_filters,
                             kernel_size=(3, 3),
                             padding='same',
                             activation='relu')

        self.conv_4 = Conv2D(filters=self.n_classes,
                             kernel_size=(1, 1),
                             padding='same',
                             activation='softmax')

        self.max_pooling = MaxPooling2D(pool_size=(2, 2),
                                        padding='same')

        self.upsampling = UpSampling2D(size=(2, 2))

        self.batch_normalization = BatchNormalization()

        self.concatenate = Concatenate(axis=-1)

        self.dropout = Dropout(rate=self.dropout_rate)

    def call(self, inputs,
                 training=True):

        include_dropout = training and self.dropout_rate == 0.0

        # downsampling path
        x0 = self.conv_0(inputs)
        x0 = self.conv_0(x0)
        x1 = self.max_pooling(x0)
        x1 = self.conv_1(x1)
        x1 = self.conv_1(x1)
        x2 = self.max_pooling(x1)
        for i in range(4):
            x2 = self.conv_2(x2)
        x3 = self.max_pooling(x2)
        for i in range(4):
            x3 = self.conv_3(x3)
        x4 = self.max_pooling(x3)
        for i in range(4):
            x4 = self.conv_3(x4)
        x5 = self.max_pooling(x4)

        # upsampling path
