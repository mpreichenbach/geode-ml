# models.py

from geode.metrics import f1, jaccard, total_accuracy
from geode.utilities import predict_raster
from numpy import unique
from os import listdir, makedirs
from os.path import isdir, join
from osgeo.gdal import Open
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, MaxPooling2D, UpSampling2D


class SegmentationModel(tf.keras.Model):

    def __init__(self, test_imagery_path: str = None,
                 test_labels_path: str = None,
                 test_predictions_path: str = None):

        super().__init__()

        self.test_metrics = {}
        self.test_imagery_path = test_imagery_path
        self.test_labels_path = test_labels_path
        self.test_predictions_path = test_predictions_path

        if set(listdir(self.test_imagery_path)) == set(listdir(self.test_labels_path)):
            self.test_filenames = listdir(self.test_imagery_path)
            if len(self.test_filenames) == 0:
                raise Exception("There is no test imagery.")
        else:
            raise Exception("The test imagery and labels must have identical filenames.")

    def compute_metrics(self, output_path: str = None) -> dict:
        """Computes various metrics on a test dataset; paired images and labels should have identical filenames.

        Args:
            output_path: the path to write a text-file of metrics.

        Returns:
             A dictionary containing various calculated metrics for each test raster.

        Raises:
            Exception: if there are no predicted rasters at test_predictions_path.
        """

        # check that there are predictions
        if len(listdir(self.test_predictions_path)) == 0:
            raise Exception("No predicted imagery has been generated.")

        # create dictionary to hold metric dictionaries
        fname_metrics = {}

        # loop through the test imagery
        for fname in self.test_filenames:
            # create metrics dictionary
            metrics_dict = {}

            # open the relevant datasets
            y_true = Open(join(self.test_labels_path, fname)).ReadAsArray()
            y_pred = Open(join(self.test_predictions_path, fname)).ReadAsArray()

            # get the label values
            labels = unique(y_true)

            # compute total accuracy
            metrics_dict["total_accuracy"] = total_accuracy(y_true, y_pred)

            # compute F1 and Jaccard scores for each label
            f1_scores = []
            jaccard_scores = []
            for label in labels:
                f1_scores.append(f1(y_true=y_true,
                                    y_pred=y_pred,
                                    pos_label=label))

                jaccard_scores.append(jaccard(y_true=y_true,
                                              y_pred=y_pred,
                                              pos_label=label))

            # add F1 and Jaccard scores to the metrics dictionary
            metrics_dict["F1"] = f1_scores
            metrics_dict["Jaccard"] = jaccard_scores

            fname_metrics[fname] = metrics_dict

        # write the dictionary to a file
        if output_path is not None:
            with open(output_path, 'w') as f:
                for key, value in fname_metrics.items():
                    f.write('%s: %s' % (key, value))

        self.metrics = fname_metrics

        return fname_metrics

    def predict_test_imagery(self, verbose=True) -> None:
        """Predicts the test imagery in the supplied path.

        Args:
            verbose: whether to print an update for each file when inference is completed.

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
                 dropout_rate: float = 0.2,
                 test_imagery_path: str = None,
                 test_labels_path: str = None,
                 test_predictions_path: str = None):

        # initialize the Model superclass
        super().__init__(test_imagery_path=test_imagery_path,
                         test_labels_path=test_labels_path,
                         test_predictions_path=test_predictions_path)

        # define attributes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        # Multiple layer versions are required because they get called on different input shapes

        # Downsampling-path convolutional layers
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

        # upsampling path

        # level 3
        u3 = self.upsampling[3](d4)
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
