# models.py

import geode.metrics as gm
from geode.utilities import predict_raster
from numpy import unique
from os import listdir, makedirs
from os.path import isdir, join
from osgeo.gdal import Open
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, MaxPooling2D, UpSampling2D


class SegmentationModel(tf.keras.Model):

    def __init__(self, test_imagery_path: str,
                 test_labels_path: str,
                 test_predictions_path: ""):

        self.metrics = {}
        self.test_imagery_path = test_imagery_path
        self.test_labels_path = test_labels_path
        self.test_predictions_path = test_predictions_path

        super().__init__()

        if set(listdir(self.test_imagery_path)) == set(listdir(self.test_labels_path)):
            self.filenames = listdir(self.test_imagery_path)
            if len(self.filenames) == 0:
                raise Exception("There is no test imagery.")
        else:
            raise Exception("The test imagery and labels must have identical filenames.")

    def compute_metrics(self) -> dict:
        """Computes various metrics on a test dataset; paired images and labels should have identical filenames.

        Returns:
             A dictionary containing various calculated metrics for each test raster.

        Raises:
            Exception: if there are no predicted rasters at test_predictions_path.
        """

        # check that there are predictions
        if len(listdir(self.test_predictions_path)) == 0:
            raise Exception("No predicted imagery has been generated.")

        # loop through the test imagery
        for fname in self.filenames:

            # open the relevant datasets
            labels = Open(join(self.test_labels_path)).ReadAsArray()
            pred = Open(join(self.test_predictions_path)).ReadAsArray()
            labels = unique(pred)

            # create dictionary to hold metrics
            metric_dict = {}


            ##### start here 

            for i in range(len(self.class_names)):
                df_precision.loc[data_name, precision_names[i]] = precision(y_true, y_pred, pos_label=i)
                df_recall.loc[data_name, recall_names[i]] = recall(y_true, y_pred, pos_label=i)
                df_jaccard.loc[data_name, jaccard_names[i]] = jaccard(y_true, y_pred, pos_label=i)
                df_f1.loc[data_name, f1_names[i]] = f1(y_true, y_pred, pos_label=i)

            # bring together all columns for one data_name
            self.metrics = pd.concat([df_precision, df_recall, df_jaccard, df_f1], axis=1)

            toc = time.perf_counter()
            metrics_time = round(toc - tic, 2)

            if verbose:
                print("Metrics for " + data_name + " generated in " + str(metrics_time) + " seconds.")

            # generate confusion tables
            tic = time.perf_counter()

            table = confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize='true')
            self.confusion_tables[data_name] = pd.DataFrame(table, index=self.class_names, columns=self.class_names)

            toc = time.perf_counter()
            confusion_time = round(toc - tic, 2)
            if verbose:
                print("Confusion table for " + data_name + " generated in " + str(confusion_time) + " seconds.")

    def predict_test_imagery(self, verbose=True) -> None:
        """Predicts the test imagery in the supplied path.

        Returns:
            None
        """

        # get filenames
        filenames = listdir(self.test_imagery_path)

        # create directory for predicted rasters
        if isdir(self.predictions_path):
            pass
        else:
            makedirs(self.predictions_path)

        # loop through the files in test_imagery_path
        for fname in filenames:
            rgb = Open(join(self.test_imagery_path, fname))

            predict_raster(input_dataset=rgb,
                           model=self,
                           output_path=join(self.test_predictions_path, fname))

            # close the input dataset
            rgb = None

            # print status if required
            if verbose:
                print("Prediction finished for", fname + ".")

class Unet(SegmentationModel):

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
