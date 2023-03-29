# models.py

from geode.metrics import f1, jaccard, total_accuracy
from geode.utilities import predict_raster
from numpy import unique
from os import listdir, makedirs
from os.path import isdir, join
from osgeo.gdal import Open
import tensorflow as tf
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, \
    UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


class SegmentationModel:

    def __init__(self):

        self.test_metrics = {}
        self.test_filenames = []
        self.model = None

    def compute_metrics(self, test_labels_path: str = None,
                        test_predictions_path: str = None,
                        output_path: str = None) -> dict:

        """Computes various metrics on a test dataset; paired images and labels should have identical filenames.

        Args:
            test_labels_path: the location of test labels;
            test_predictions_path: the location at which to save model predictions;
            output_path: the path to write a text-file of metrics.

        Returns:
             A dictionary containing various calculated metrics for each test raster.

        Raises:
            Exception: if there are no predicted rasters at test_predictions_path.
        """

        # check that there are predictions
        if len(listdir(test_predictions_path)) == 0:
            raise Exception("No predicted imagery has been generated.")

        # create dictionary to hold metric dictionaries
        fname_metrics = {}

        # loop through the test imagery
        for fname in self.test_filenames:
            # create metrics dictionary
            metrics_dict = {}

            # open the relevant datasets
            y_true = Open(join(test_labels_path, fname)).ReadAsArray()
            y_pred = Open(join(test_predictions_path, fname)).ReadAsArray()

            # get the label values
            labels = unique(y_true)

            # compute total accuracy
            metrics_dict['total_accuracy'] = total_accuracy(y_true, y_pred)

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
            metrics_dict['F1'] = f1_scores
            metrics_dict['Jaccard'] = jaccard_scores

            fname_metrics[fname] = metrics_dict

        # write the dictionary to a file
        if output_path is not None:
            with open(output_path, 'w') as f:
                for key, value in fname_metrics.items():
                    f.write('%s: %s' % (key, value))

        self.test_metrics = fname_metrics

        return fname_metrics

    def predict_test_imagery(self, test_imagery_path: str = None,
                             test_labels_path: str = None,
                             test_predictions_path: str = None,
                             verbose=True) -> None:
        """Predicts the test imagery in the supplied path.

        Args:
            test_imagery_path: the location of input test imagery;
            test_labels_path: the location of test labels;
            test_predictions_path: the location at which to save model predictions;
            verbose: whether to print an update for each file when inference is completed.

        Returns:
            None

        Raises:
            Exception: if any of the input paths are None;
            Exception: if no test files exist at the supplied paths.
        """

        # check that input paths are supplied
        if test_imagery_path is None or test_labels_path is None or test_predictions_path is None:
            raise Exception("One of the required path arguments has not been supplied.")

        # check that test imagery exists and has correctly named labels
        if set(listdir(test_imagery_path)) == set(listdir(test_labels_path)):
            self.test_filenames = listdir(test_imagery_path)
            if len(self.test_filenames) == 0:
                raise Exception("There is no test imagery.")
        else:
            raise Exception("The test imagery and labels must have identical filenames.")

        # get filenames
        filenames = listdir(test_imagery_path)

        # create directory for predicted rasters
        if isdir(test_predictions_path):
            pass
        else:
            makedirs(test_predictions_path)

        # loop through the files in test_imagery_path
        for fname in filenames:
            rgb = Open(join(test_imagery_path, fname))

            predict_raster(input_dataset=rgb,
                           model=self.model,
                           output_path=join(test_predictions_path, fname))

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

        # initialize the superclass
        super().__init__()

        # define attributes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        # ensure that n_classes >= 2
        if self.n_classes < 2:
            raise Exception("Number of classes must at least 2.")

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

        # initialize the superclass
        super().__init__()

        # define attributes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

        # ensure that n_classes >= 2
        if self.n_classes < 2:
            raise Exception("Number of classes must at least 2.")

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
