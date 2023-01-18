# models.py


import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, MaxPooling2D, UpSampling2D



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

        # Multiple layer versions are required because they get called on different input shapes

        # Dowsampling-path convolutional layers
        self.conv_down_0 = [Conv2D(filters=self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(2)]

        self.conv_down_1 = [Conv2D(filters=2 * self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(2)]

        self.conv_down_2 = [Conv2D(filters=4 * self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(4)]

        self.conv_down_3 = [Conv2D(filters=8 * self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(4)]

        self.conv_down_4 = [Conv2D(filters=8 * self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(4)]

        self.conv_down_5 = [Conv2D(filters=8 * self.n_filters,
                            kernel_size=(3, 3),
                            padding='same',
                            activation='relu') for i in range(4)]

        self.conv_up_4 = [Conv2D(filters=8 * self.n_filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu') for i in range(4)]

        self.conv_up_3 = [Conv2D(filters=8 * self.n_filters,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu') for i in range(4)]

        self.conv_up_2 = [Conv2D(filters=4 * self.n_filters,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu') for i in range(4)]

        self.conv_up_1 = [Conv2D(filters=2 * self.n_filters,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu') for i in range(2)]

        self.conv_up_0 = [Conv2D(filters=8 * self.n_filters,
                                 kernel_size=(3, 3),
                                 padding='same',
                                 activation='relu') for i in range(2)]

        self.conv_final = Conv2D(filters=self.n_classes,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 activation='softmax')

        # Compute how many dropout and batch-normalization layers are needed
        n_do_bn = len(self.conv_down_0 + self.conv_up_0 +
                      self.conv_down_1 + self.conv_up_1 +
                      self.conv_down_2 + self.conv_up_2 +
                      self.conv_down_3 + self.conv_up_3 +
                      self.conv_down_4 + self.conv_up_4 +
                      self.conv_down_5)

        # Max-pooling layers
        self.max_pooling = [MaxPooling2D(pool_size=(2, 2),
                                         padding='same') for i in range(5)]

        # Upsampling layers
        self.upsampling = [UpSampling2D(size=(2, 2)) for i in range(5)]

        # Batch normalization layers
        self.batch_normalization = [BatchNormalization() for i in range(n_do_bn)]

        # Dropout layers
        self.dropout = [Dropout(rate=self.dropout_rate) for i in range(n_do_bn)]

        # Concatenate layers
        self.concatenate = [Concatenate(axis=-1) for i in range(5)]

    def call(self, input_tensor,
             training=True):

        include_dropout = training and self.dropout_rate == 0.0
        conv_counter = 0

        # downsampling path

        # level 0
        d0 = input_tensor
        for i in range(2):
            d0 = self.conv_down_0[i](d0)
            d0 = self.dropout[conv_counter](d0) if include_dropout else d0
            d0 = self.batch_normalization[conv_counter](d0)
            conv_counter += 1

        # level 1
        d1 = self.max_pooling[0](d0)
        for i in range(2):
            d1 = self.conv_down_1[i](d1)
            d1 = self.dropout[conv_counter](d1) if include_dropout else d1
            d1 = self.batch_normalization[conv_counter](d1)
            conv_counter += 1

        # level 2
        d2 = self.max_pooling[1](d1)
        for i in range(4):
            d2 = self.conv_down_2[i](d2)
            d2 = self.dropout[conv_counter](d2) if include_dropout else d2
            d2 = self.batch_normalization[conv_counter](d2)
            conv_counter += 1

        # level 3
        d3 = self.max_pooling[2](d2)
        for i in range(4):
            d3 = self.conv_down_3[i](d3)
            d3 = self.dropout[conv_counter](d3) if include_dropout else d3
            d3 = self.batch_normalization[conv_counter](d3)
            conv_counter += 1

        # level 4
        d4 = self.max_pooling[3](d3)
        for i in range(4):
            d4 = self.conv_down_4[i](d4)
            d4 = self.dropout[conv_counter](d4) if include_dropout else d4
            d4 = self.batch_normalization[conv_counter](d4)
            conv_counter += 1

        # level 5
        d5 = self.max_pooling[4](d4)
        for i in range(4):
            d5 = self.conv_down_5[i](d5)
            d5 = self.dropout[conv_counter](d5) if include_dropout else d5
            d5 = self.batch_normalization[conv_counter](d5)
            conv_counter += 1

        # upsampling path

        # level 4
        u4 = self.upsampling[4](d5)
        u4 = self.concatenate[4]([u4, d4])
        for i in range(4):
            u4 = self.conv_up_4[i](u4)
            u4 = self.dropout[conv_counter](u4) if include_dropout else u4
            u4 = self.batch_normalization[conv_counter](u4)
            conv_counter += 1

        # level 3
        u3 = self.upsampling[3](u4)
        u3 = self.concatenate[3]([u3, d3])
        for i in range(4):
            u3 = self.conv_up_3[i](u3)
            u3 = self.dropout[conv_counter](u3) if include_dropout else u3
            u3 = self.batch_normalization[conv_counter](u3)
            conv_counter += 1

        # level 2
        u2 = self.upsampling[2](u3)
        u2 = self.concatenate[2]([u2, d2])
        for i in range(4):
            u2 = self.conv_up_2[i](u2)
            u2 = self.dropout[conv_counter](u2) if include_dropout else u2
            u2 = self.batch_normalization[conv_counter](u2)
            conv_counter += 1

        # level 1
        u1 = self.upsampling[1](u2)
        u1 = self.concatenate[1]([u1, d1])
        for i in range(2):
            u1 = self.conv_up_1[i](u1)
            u1 = self.dropout[conv_counter](u1) if include_dropout else u1
            u1 = self.batch_normalization[conv_counter](u1)
            conv_counter += 1

        # level 0
        u0 = self.upsampling[0](u1)
        u0 = self.concatenate[0]([u0, d0])
        for i in range(2):
            u0 = self.conv_up_0[i](u0)
            u0 = self.dropout[conv_counter](u0) if include_dropout else u0
            u0 = self.batch_normalization[conv_counter](u0)
            conv_counter += 1

        output = self.conv_final(u0)

        return output

