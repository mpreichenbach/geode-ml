# models.py

from geode.metrics import f1, jaccard, total_accuracy
from geode.utilities import predict_raster
from numpy import unique
from os import listdir, makedirs
from os.path import isdir, join
from osgeo.gdal import Open
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, Dropout, Input, MaxPooling2D, UpSampling2D


class SegmentationModel:

    def __init__(self):

        super().__init__()

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


class Unet(SegmentationModel):

    def __init__(self, n_channels: int = 3,
                 n_classes: int = 2,
                 n_filters: int = 16,
                 dropout_rate: float = 0.3):

        # initialize the superclass
        super().__init__()

        # define attributes
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate

    def compile_model(self, loss: tf.keras.losses.Loss = 'sparse_categorical_crossentropy',
                      learning_rate: float = 0.0001) -> None:

        """Returns a model object, compiled with the provided loss and optimizer. Additionally, this sets the self.model
        attribute with the compiled model.

        Args:
            loss: the loss function to use during training;
            learning_rate: the starting learning rate for the Adam optimizer.

        Returns:
            None"""

        include_dropout = (self.dropout_rate > 0.0)

        # build the model graph

        # level 0
        inputs = Input(shape=(None, None, self.n_channels), dtype=tf.float32)
        d0 = Conv2D(filters=self.n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu')(inputs)
        d0 = Dropout(rate=self.dropout_rate)(d0) if include_dropout else d0
        d0 = BatchNormalization()(d0)
        d0 = Conv2D(filters=self.n_filters,
                    kernel_size=(3, 3),
                    padding='same',
                    activation='relu')(d0)
        d0 = Dropout(rate=self.dropout_rate)(d0) if include_dropout else d0
        d0 = BatchNormalization()(d0)

        # level 1
        d1 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d0)
        for i in range(2):
            d1 = Conv2D(filters=2 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(d1)
            d1 = Dropout(rate=self.dropout_rate)(d1) if include_dropout else d1
            d1 = BatchNormalization()(d1)

        # level 2
        d2 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d1)
        for i in range(4):
            d2 = Conv2D(filters=4 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(d2)
            d2 = Dropout(rate=self.dropout_rate)(d2) if include_dropout else d2
            d2 = BatchNormalization()(d2)

        # level 3
        d3 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d2)
        for i in range(4):
            d3 = Conv2D(filters=8 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(d3)
            d3 = Dropout(rate=self.dropout_rate)(d3) if include_dropout else d3
            d3 = BatchNormalization()(d3)

        # level 4
        d4 = MaxPooling2D(pool_size=(2, 2),
                          padding='same')(d3)
        for i in range(4):
            d4 = Conv2D(filters=8 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(d4)
            d4 = Dropout(rate=self.dropout_rate)(d4) if include_dropout else d4
            d4 = BatchNormalization()(d4)

        # upsampling path

        # level 3
        u3 = UpSampling2D(size=(2, 2))(d4)
        u3 = Concatenate(axis=-1)([u3, d3])
        for i in range(4):
            u3 = Conv2D(filters=8 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(u3)
            u3 = Dropout(rate=self.dropout_rate)(u3) if include_dropout else u3
            u3 = BatchNormalization()(u3)

        # level 2
        u2 = UpSampling2D(size=(2, 2))(u3)
        u2 = Concatenate(axis=-1)([u2, d2])
        for i in range(4):
            u2 = Conv2D(filters=4 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(u2)
            u2 = Dropout(rate=self.dropout_rate)(u2) if include_dropout else u2
            u2 = BatchNormalization()(u2)

        # level 1
        u1 = UpSampling2D(size=(2, 2))(u2)
        u1 = Concatenate(axis=-1)([u1, d1])
        for i in range(2):
            u1 = Conv2D(filters=2 * self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(u1)
            u1 = Dropout(rate=self.dropout_rate)(u1) if include_dropout else u1
            u1 = BatchNormalization()(u1)

        # level 0
        u0 = UpSampling2D(size=(2, 2))(u1)
        u0 = Concatenate(axis=-1)([u0, d0])
        for i in range(2):
            u0 = Conv2D(filters=self.n_filters,
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(u0)
            u0 = Dropout(rate=self.dropout_rate)(u0) if include_dropout else u0
            u0 = BatchNormalization()(u0)

        outputs = Conv2D(filters=self.n_classes,
                         kernel_size=(1, 1),
                         padding='same',
                         activation='softmax')(u0)

        # create the model object
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # compile the model
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        self.model = model
