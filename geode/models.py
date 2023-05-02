# models.py

import geode.metrics as gm
from geode.utilities import predict_raster
from numpy import mean, unique
from os import listdir, makedirs
from os.path import isdir, join
from osgeo.gdal import Open
import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, \
    UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class SegmentationModel:
    """A class for defining and testing semantic segmentation models.

    Attributes:
        test_imagery_path: the directory containing the test imagery;
        test_labels_path: the directory containing the true labels;
        test_predictions_path: the directory containing the predicted labels;
        data_names: a list of imagery names (usually regions before tiling was applied);
        test_metrics: the dictionary which stores computed metrics;
        test_imagery_names: the files which make up the test imagery set;
        model: the model object."""

    def __init__(self, test_imagery_path: str = None,
                 test_labels_path: str = None,
                 test_predictions_path: str = None,
                 data_names: list = None):

        self.test_imagery_path = str(test_imagery_path)
        self.test_labels_path = str(test_labels_path)
        self.test_predictions_path = str(test_predictions_path)
        self.data_names = str(data_names)
        self.test_metrics = {}
        self.model = None

        if test_imagery_path is not None:
            self.test_imagery_names = listdir(test_imagery_path)

        if len(self.test_imagery_names) == 0:
            raise Exception("No test imagery has been supplied.")

        if test_labels_path is not None:
            self.test_labels_name = listdir(test_labels_path)

        if len(self.test_imagery_names) != len(self.test_imagery_names):
            raise Exception("There are a different number of test imagery and label files.")

        if set(self.test_imagery_names) != set(self.test_labels_name):
            raise Exception("Test imagery and label file names do not match.")

    def compute_metrics(self, output_path: str = None,
                        pos_label: int = 1,
                        verbose: bool = True) -> dict:

        """Computes various metrics on a test dataset; paired images and labels should have identical filenames. Both
        Jaccard and F1 metrics assume binary labeling.

        Args:
            output_path: the path to write a text-file of metrics;
            pos_label: the label for which to compute metrics;
            verbose: whether to print progress.

        Returns:
             A dictionary containing various calculated metrics for each test raster.

        Raises:
            Exception: if there are no predicted rasters at test_predictions_path;
        """

        # coerce arguments to correct type
        output_path = str(output_path)
        pos_label = int(pos_label)
        verbose = bool(verbose)

        # check that there are predictions
        if len(listdir(self.test_predictions_path)) == 0:
            raise Exception("No predicted imagery has been generated.")

        # create dictionary to hold metric dictionaries
        dname_metrics = {}

        # get the test_filenames:
        filenames = listdir(self.test_labels_path)

        # loop through the test imagery
        for dname in self.data_names:
            if verbose:
                print(dname)

            # get the relevant subset
            sub_filenames = [x for x in filenames if dname in x]

            # create metrics dictionary
            metrics_dict = {}

            # create lists for each metric
            true_positive_scores = []
            false_positive_scores = []
            false_negative_scores = []
            f1_scores = []
            jaccard_scores = []
            acc_scores = []

            # loop through the test subset
            for fname in sub_filenames:
                if verbose:
                    print(fname)

                # open the relevant datasets
                y_true = Open(join(self.test_labels_path, fname)).ReadAsArray()
                y_pred = Open(join(self.test_predictions_path, fname)).ReadAsArray()

                # compute metrics
                true_positive_scores.append(gm.true_positives(y_true=y_true,
                                                              y_pred=y_pred,
                                                              pos_label=pos_label))

                false_positive_scores.append(gm.false_positives(y_true=y_true,
                                                                y_pred=y_pred,
                                                                pos_label=pos_label))

                false_negative_scores.append(gm.false_negatives(y_true=y_true,
                                                                y_pred=y_pred,
                                                                pos_label=pos_label))

                f1_scores.append(gm.f1(y_true=y_true,
                                       y_pred=y_pred,
                                       pos_label=pos_label))

                jaccard_scores.append(gm.jaccard(y_true=y_true,
                                                 y_pred=y_pred,
                                                 pos_label=pos_label))

                precision_scores.append(gm.precision(y_true=y_true,
                                                     y_pred=y_pred,
                                                     pos_label=pos_label))

                recall_scores.append(gm.recall(y_true=y_true,
                                               y_pred=y_pred,
                                               pos_label=pos_label))

                acc_scores.append(gm.total_accuracy(y_true=y_true,
                                                    y_pred=y_pred))

            # add scores to the metrics dictionary
            metrics_dict['true_positives'] = mean(true_positive_scores)
            metrics_dict['false_positives'] = mean(false_positive_scores)
            metrics_dict['false_negatives'] = mean(false_negative_scores)
            metrics_dict['f1'] = mean(f1_scores)
            metrics_dict['jaccard'] = mean(jaccard_scores)
            metrics_dict['precision'] = mean(precision_scores)
            metrics_dict['recall'] = mean(recall_scores)
            metrics_dict['accuracy'] = mean(acc_scores)

            dname_metrics[dname] = metrics_dict

        # write the dictionary to a file
        if output_path is not None:
            with open(output_path, 'w') as f:
                for key, value in dname_metrics.items():
                    f.write('%s: %s' % (key, value))

        self.test_metrics = dname_metrics

        return dname_metrics

    def predict_test_imagery(self, tile_dim: int = 512,
                             verbose: bool = True) -> None:

        """Predicts the test imagery in the supplied path.

        Args:
            tile_dim: the square dimensions of the tile to predict;
            verbose: whether to print an update for each file when inference is completed.

        Returns:
            None

        Raises:
            TypeError: if verbose is not boolean;
        """

        # check for the correct types
        tile_dim = int(tile_dim)
        if not isinstance(verbose, bool):
            raise TypeError("Argument verbose must be boolean.")

        # get filenames
        filenames = listdir(self.test_imagery_path)

        # create directory for predicted rasters
        if isdir(self.test_predictions_path):
            pass
        else:
            makedirs(self.test_predictions_path)

        # loop through the files in test_imagery_path
        for fname in filenames:
            rgb = Open(join(self.test_imagery_path, fname))

            predict_raster(input_dataset=rgb,
                           model=self.model,
                           output_path=join(self.test_predictions_path, fname),
                           tile_dim=tile_dim)

            # close the input dataset
            rgb = None

            # print status if required
            if verbose:
                print("Prediction finished for", fname + ".")


class VGG19Unet(SegmentationModel):

    def __init__(self, n_channels: int = 3,
                 n_classes: int = 2,
                 n_filters: int = 16,
                 dropout_rate: float = 0.3,
                 rescale_factor: float = 1 / 255,
                 include_residual: bool = False):

        """Instantiates the Unet architecture, with mirrored VGG19 architectures for the up- and down-sampling paths.

        Attributes:
            n_channels: the number of channels in the input imagery;
            n_classes: the number of classes to predict;
            n_filters: the number of convolutional filters in the first layer;
            dropout_rate: the proportion of nodes to turn off for each inference step during training.

        Raises:
            ValueError: if n_channels is less than 1;
            ValueError: if n_classes is less than 2;
            ValueError: if n_filters is less than 1;
            ValueError: if dropout_rate is not between 0.0 and 1.0;
            TypeError:  if include_residual is not boolean.
        """

        # initialize the superclass
        super().__init__()

        # coerce arguments to correct types and define attributes
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.n_filters = int(n_filters)
        self.dropout_rate = float(dropout_rate)

        # perform type-checking
        if self.n_channels < 1:
            raise ValueError("The argument n_channels must be greater than or equal to one.")
        if self.n_classes < 2:
            raise ValueError("The argument n_classes must be greater than or equal to 2.")
        if self.n_filters < 1:
            raise ValueError("The argument n_filters must be at least 1.")
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("The argument dropout_rate must be between 0.0 and 1.0, inclusive.")
        if not isinstance(include_residual, bool):
            raise TypeError("The argument include_residual must be boolean.")

        # define the layers and model
        include_dropout = (self.dropout_rate > 0.0)

        def conv_block(input_tensor,
                       filters):

            conv = Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu')(input_tensor)
            dropout = Dropout(rate=self.dropout_rate)(conv) if include_dropout else conv
            batch_norm = BatchNormalization()(dropout)

            return batch_norm

        # level 0
        inputs = Input(shape=(None, None, self.n_channels), dtype=tf.float32)
        d0 = Rescaling(scale=rescale_factor)(inputs)
        d0_conv_1 = conv_block(d0, filters=self.n_filters)
        d0_conv_2 = conv_block(d0_conv_1, filters=self.n_filters)
        d0_out = Add()([d0_conv_1, d0_conv_2]) if include_residual else d0_conv_2

        # level 1
        d1 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d0_out)
        d1_conv_1 = conv_block(d1, filters=2 * self.n_filters)
        d1_conv_2 = conv_block(d1_conv_1, filters=2 * self.n_filters)
        d1_out = Add()([d1_conv_1, d1_conv_2]) if include_residual else d1_conv_2

        # level 2
        d2 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d1_out)
        d2_conv_1 = conv_block(d2, filters=4 * self.n_filters)
        d2_conv_2 = conv_block(d2_conv_1, filters=4 * self.n_filters)
        d2_conv_2 = Add()([d2_conv_1, d2_conv_2]) if include_residual else d2_conv_2
        d2_conv_3 = conv_block(d2_conv_2, filters=4 * self.n_filters)
        d2_conv_3 = Add()([d2_conv_2, d2_conv_3]) if include_residual else d2_conv_3
        d2_conv_4 = conv_block(d2_conv_3, filters=4 * self.n_filters)
        d2_out = Add()([d2_conv_3, d2_conv_4]) if include_residual else d2_conv_4

        # level 3
        d3 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d2_out)
        d3_conv_1 = conv_block(d3, filters=8 * self.n_filters)
        d3_conv_2 = conv_block(d3_conv_1, filters=8 * self.n_filters)
        d3_conv_2 = Add()([d3_conv_1, d3_conv_2]) if include_residual else d3_conv_2
        d3_conv_3 = conv_block(d3_conv_2, filters=8 * self.n_filters)
        d3_conv_3 = Add()([d3_conv_2, d3_conv_3]) if include_residual else d3_conv_3
        d3_conv_4 = conv_block(d3_conv_3, filters=8 * self.n_filters)
        d3_out = Add()([d3_conv_3, d3_conv_4]) if include_residual else d3_conv_4

        # level 4
        d4 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d3_out)
        d4_conv_1 = conv_block(d4, filters=8 * self.n_filters)
        d4_conv_2 = conv_block(d4_conv_1, filters=8 * self.n_filters)
        d4_conv_2 = Add()([d4_conv_1, d4_conv_2]) if include_residual else d4_conv_2
        d4_conv_3 = conv_block(d4_conv_2, filters=8 * self.n_filters)
        d4_conv_3 = Add()([d4_conv_2, d4_conv_3]) if include_residual else d4_conv_3
        d4_conv_4 = conv_block(d4_conv_3, filters=8 * self.n_filters)
        d4_out = Add()([d4_conv_3, d4_conv_4]) if include_residual else d4_conv_4

        # upsampling path

        # level 3
        u3 = UpSampling2D(size=(2, 2))(d4_out)
        u3 = Concatenate(axis=-1)([u3, d3_out])
        u3_conv_1 = conv_block(u3, filters=8 * self.n_filters)
        u3_conv_2 = conv_block(u3_conv_1, filters=8 * self.n_filters)
        u3_conv_2 = Add()([u3_conv_1, u3_conv_2]) if include_residual else u3_conv_2
        u3_conv_3 = conv_block(u3_conv_2, filters=8 * self.n_filters)
        u3_conv_3 = Add()([u3_conv_2, u3_conv_3]) if include_residual else u3_conv_3
        u3_conv_4 = conv_block(u3_conv_3, filters=8 * self.n_filters)
        u3_out = Add()([u3_conv_3, u3_conv_4]) if include_residual else u3_conv_4

        # level 2
        u2 = UpSampling2D(size=(2, 2))(u3_out)
        u2 = Concatenate(axis=-1)([u2, d2_out])
        u2_conv_1 = conv_block(u2, filters=4 * self.n_filters)
        u2_conv_2 = conv_block(u2_conv_1, filters=4 * self.n_filters)
        u2_conv_2 = Add()([u2_conv_1, u2_conv_2]) if include_residual else u2_conv_2
        u2_conv_3 = conv_block(u2_conv_2, filters=4 * self.n_filters)
        u2_conv_3 = Add()([u2_conv_2, u2_conv_3]) if include_residual else u2_conv_3
        u2_conv_4 = conv_block(u2_conv_3, filters=4 * self.n_filters)
        u2_out = Add()([u2_conv_3, u2_conv_4]) if include_residual else u2_conv_4

        # level 1
        u1 = UpSampling2D(size=(2, 2))(u2_out)
        u1 = Concatenate(axis=-1)([u1, d1_out])
        u1_conv_1 = conv_block(u1, filters=2 * self.n_filters)
        u1_conv_2 = conv_block(u1_conv_1, filters=2 * self.n_filters)
        u1_conv_2 = Add()([u1_conv_1, u1_conv_2]) if include_residual else u1_conv_2
        u1_conv_3 = conv_block(u1_conv_2, filters=2 * self.n_filters)
        u1_conv_3 = Add()([u1_conv_2, u1_conv_3]) if include_residual else u1_conv_3
        u1_conv_4 = conv_block(u1_conv_3, filters=2 * self.n_filters)
        u1_out = Add()([u1_conv_3, u1_conv_4]) if include_residual else u1_conv_4

        # level 0
        u0 = UpSampling2D(size=(2, 2))(u1_out)
        u0 = Concatenate(axis=-1)([u0, d0_out])
        u0_conv_1 = conv_block(u0, filters=self.n_filters)
        u0_conv_2 = conv_block(u0_conv_1, filters=self.n_filters)
        u0_conv_2 = Add()([u0_conv_1, u0_conv_2]) if include_residual else u0_conv_2
        u0_conv_3 = conv_block(u0_conv_2, filters=self.n_filters)
        u0_conv_3 = Add()([u0_conv_2, u0_conv_3]) if include_residual else u0_conv_3
        u0_conv_4 = conv_block(u0_conv_3, filters=self.n_filters)
        u0_out = Add()([u0_conv_3, u0_conv_4]) if include_residual else u0_conv_4

        outputs = Conv2D(filters=self.n_classes,
                         kernel_size=(1, 1),
                         padding='same',
                         activation='softmax')(u0_out)

        # create the model object
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self, loss=None,
                learning_rate: float = 0.001) -> None:

        """Compiles the model attribute with the given loss function, and an Adam optimizer with the provided learning
        rate.

        Args:
            loss: the loss function to use during training;
            learning_rate: the starting learning rate for the Adam optimizer.

        Returns:
            None
        """

        # coerce arguments to correct type
        learning_rate = float(learning_rate)

        # compile the model
        self.model.compile(loss=loss,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


class Unet(SegmentationModel):

    def __init__(self, n_channels: int = 3,
                 n_classes: int = 2,
                 n_filters: int = 16,
                 dropout_rate: float = 0.3,
                 rescale_factor: float = 1 / 255,
                 include_residual: bool = False):

        """Instantiates the Unet architecture, with filters doubling in each level.

        Attributes:
            n_channels: the number of channels in the input imagery;
            n_classes: the number of classes to predict;
            n_filters: the number of convolutional filters in the first layer;
            dropout_rate: the proportion of nodes to turn off for each inference step during training.

        Raises:
            ValueError: if n_channels is less than 1;
            ValueError: if n_classes is less than 2;
            ValueError: if n_filters is less than 1;
            ValueError: if dropout_rate is not between 0.0 and 1.0;
            TypeError:  if include_residual is not boolean.
        """

        # initialize the superclass
        super().__init__()

        # coerce arguments to correct types and define attributes
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.n_filters = int(n_filters)
        self.dropout_rate = float(dropout_rate)

        # perform type-checking
        if self.n_channels < 1:
            raise ValueError("The argument n_channels must be greater than or equal to one.")
        if self.n_classes < 2:
            raise ValueError("The argument n_classes must be greater than or equal to 2.")
        if self.n_filters < 1:
            raise ValueError("The argument n_filters must be at least 1.")
        if self.dropout_rate < 0.0 or self.dropout_rate > 1.0:
            raise ValueError("The argument dropout_rate must be between 0.0 and 1.0, inclusive.")
        if not isinstance(include_residual, bool):
            raise TypeError("The argument include_residual must be boolean.")

        include_dropout = (self.dropout_rate > 0.0)

        def conv_block(input_tensor,
                       filters):

            conv = Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu')(input_tensor)
            dropout = Dropout(rate=self.dropout_rate)(conv) if include_dropout else conv
            batch_norm = BatchNormalization()(dropout)

            return batch_norm

        # build the model graph

        # level 0
        inputs = Input(shape=(None, None, self.n_channels), dtype=tf.float32)
        d0 = Rescaling(scale=rescale_factor)(inputs)
        d0_conv_1 = conv_block(d0, filters=self.n_filters)
        d0_conv_2 = conv_block(d0_conv_1, filters=self.n_filters)
        d0_out = Add()([d0_conv_1, d0_conv_2]) if include_residual else d0_conv_2

        # level 1
        d1 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d0_out)
        d1_conv_1 = conv_block(d1, filters=2 * self.n_filters)
        d1_conv_2 = conv_block(d1_conv_1, filters=2 * self.n_filters)
        d1_out = Add()([d1_conv_1, d1_conv_2]) if include_residual else d1_conv_2

        # level 2
        d2 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d1_out)
        d2_conv_1 = conv_block(d2, filters=4 * self.n_filters)
        d2_conv_2 = conv_block(d2_conv_1, filters=4 * self.n_filters)
        d2_out = Add()([d2_conv_1, d2_conv_2]) if include_residual else d2_conv_2

        # level 3
        d3 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d2_out)
        d3_conv_1 = conv_block(d3, filters=8 * self.n_filters)
        d3_conv_2 = conv_block(d3_conv_1, filters=8 * self.n_filters)
        d3_out = Add()([d3_conv_1, d3_conv_2]) if include_residual else d3_conv_2

        # level 4
        d4 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d3_out)
        d4_conv_1 = conv_block(d4, filters=16 * self.n_filters)
        d4_conv_2 = conv_block(d4_conv_1, filters=16 * self.n_filters)
        d4_out = Add()([d4_conv_1, d4_conv_2]) if include_residual else d4_conv_2

        # upsampling path

        # level 3
        u3 = UpSampling2D(size=(2, 2))(d4_out)
        u3 = Concatenate(axis=-1)([u3, d3_out])
        u3_conv_1 = conv_block(u3, filters=8 * self.n_filters)
        u3_conv_2 = conv_block(u3_conv_1, filters=8 * self.n_filters)
        u3_out = Add()([u3_conv_1, u3_conv_2]) if include_residual else u3_conv_2

        # level 2
        u2 = UpSampling2D(size=(2, 2))(u3_out)
        u2 = Concatenate(axis=-1)([u2, d2_out])
        u2_conv_1 = conv_block(u2, filters=4 * self.n_filters)
        u2_conv_2 = conv_block(u2_conv_1, filters=4 * self.n_filters)
        u2_out = Add()([u2_conv_1, u2_conv_2]) if include_residual else u2_conv_2

        # level 1
        u1 = UpSampling2D(size=(2, 2))(u2_out)
        u1 = Concatenate(axis=-1)([u1, d1_out])
        u1_conv_1 = conv_block(u1, filters=2 * self.n_filters)
        u1_conv_2 = conv_block(u1_conv_1, filters=2 * self.n_filters)
        u1_out = Add()([u1_conv_1, u1_conv_2]) if include_residual else u1_conv_2

        # level 0
        u0 = UpSampling2D(size=(2, 2))(u1_out)
        u0 = Concatenate(axis=-1)([u0, d0_out])
        u0_conv_1 = conv_block(u0, filters=self.n_filters)
        u0_conv_2 = conv_block(u0_conv_1, filters=self.n_filters)
        u0_out = Add()([u0_conv_1, u0_conv_2]) if include_residual else u0_conv_2

        outputs = Conv2D(filters=self.n_classes,
                         kernel_size=(1, 1),
                         padding='same',
                         activation='softmax')(u0_out)

        # create the model object
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    def compile(self, loss=None,
                learning_rate: float = 0.001) -> None:

        """Compiles the model attribute with the given loss function, and an Adam optimizer with the provided learning
        rate.

        Args:
            loss: the loss function to use during training;
            learning_rate: the starting learning rate for the Adam optimizer.

        Returns:
            None
        """

        # compile the model
        self.model.compile(loss=loss,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
