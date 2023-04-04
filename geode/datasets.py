# datasets.py

from geode.utilities import convert_labels_to_one_hots, rasterize_polygon_layer, resample_dataset, tile_raster_pair
from numpy import expand_dims, moveaxis
from numpy.testing import assert_allclose
from numpy.random import shuffle
from os import listdir, mkdir
from os.path import isdir, join, splitext
from osgeo import gdal, ogr
import tensorflow as tf


class SegmentationDataset:

    """A class for defining and manipulating semantic segmentation datasets.

    Attributes:
        source_path: the directory containing source imagery;
        polygons_path: the directory containing subdirectories of polygon shapefiles;
        labels_path: the directory to contain label rasters;
        tiles_path: the directory to contain the imagery/labels subdirectory pairs for tiles;
        n_channels: the number of channels in the source_imagery;
        tile_dimension: the length/width of tiles;
        source_image_names: the filenames of source imagery;
        data_names: the source imagery names without filename extensions;
    """

    def __init__(self, source_path: str = '',
                 polygons_path: str = '',
                 labels_path: str = '',
                 tiles_path: str = '',
                 n_channels: int = 3,
                 tile_dimension: int = 0):

        """Args:
            source_path: the directory containing the source imagery;
            polygons_path: the directory containing polygons;
            labels_path: the directory containing the label rasters;
            n_channels: the number of channels in the source imagery;
            tile_dimension: the length/width of tiles;

        Raises:
            ValueError: if source_path is the empty string;
            Exception: if the source_path is empty;
            ValueError: if n_channels is negative;
            ValueError: if tile_dimension is negative.
        """

        # attempt to coerce arguments to correct type, and set attributes
        self.source_path = str(source_path)
        self.polygons_path = str(polygons_path)
        self.labels_path = str(labels_path)
        self.tiles_path = str(tiles_path)
        self.n_channels = int(n_channels)
        self.tile_dimension = int(tile_dimension)

        # check argument values
        if source_path != '':
            self.source_image_names = listdir(source_path)
            self.data_names = [splitext(x)[0] for x in self.source_image_names]
            if len(self.source_image_names) == 0:
                raise Exception('The source_path is empty.')

        if self.n_channels <= 0:
            raise ValueError("The argument n_channels must be a positive integer.")

        if self.tile_dimension <= 0:
            raise ValueError("The argument tile_dimension must be a postiive integer.")

    def check_polygons(self) -> None:

        """Checks whether the polygon data has been set, and matches the names of the source imagery.

        Returns:
            None

        Raises:
            ValueError: if polygons_path has not been set;
            Exception: if polygons_path is empty;
            Exception: if polygon data folder names do not match source imagery names;
            Exception: if a polygon data folder does not contain a shapefile.
        """

        if self.polygons_path == "":
            raise ValueError("The attribute polygons_path cannot be the empty string.")
        elif len(listdir(self.polygons_path)) == 0:
            raise Exception("The polygons_path is empty.")
        elif set([splitext(x)[0] for x in listdir(self.source_path)]) != set(listdir(self.polygons_path)):
            raise Exception("Source imagery names do not match polygon data names.")
        else:
            for directory in listdir(self.polygons_path):
                filenames = listdir(join(self.polygons_path, directory))
                shapefiles = [x for x in filenames if splitext(x)[1] == '.shp']
                if len(shapefiles) != 1:
                    raise Exception("The polygons data directories must have exactly one shapefile.")

    def check_labels(self) -> None:

        """Checks whether the label rasters have been generated, and match the names of the source imagery.

        Returns:
            None

        Raises:
            ValueError: if labels_path has not been set;
            Exception: if labels_path is empty;
            Exception: if label raster names do not match source imagery names;
            Exception: if source/raster width does not match for a particular pair;
            Exception: if source/raster height does not match for a particular pair;
            Exception: if source/raster projections do not match for a particular pair.
        """

        if self.labels_path == "":
            raise ValueError("The attribute labels_path cannot be the empty string.")
        elif len(listdir(self.labels_path)) == 0:
            raise Exception("The labels_path is empty.")
        elif set(listdir(self.source_path)) != set(listdir(self.labels_path)):
            raise Exception("Source imagery names do not match label raster names.")
        else:
            for filename in self.source_image_names:
                source_dataset = gdal.Open(join(self.source_path, filename))
                label_dataset = gdal.Open(join(self.labels_path, filename))

                # we use a numpy unittest to determine if the geotransforms are almost equal:
                assert_allclose(source_dataset.GetGeoTransform(), label_dataset.GetGeoTransform())

                # check other metadata to see if it matches
                if source_dataset.RasterXSize != label_dataset.RasterXSize:
                    raise Exception("Raster x-dimensions do not match for " + filename + " pair.")
                elif source_dataset.RasterYSize != label_dataset.RasterYSize:
                    raise Exception("Raster y-dimensions do not match for " + filename + " pair.")

    def check_tiles(self) -> None:

        """Checks whether the tiles have been correctly generated.

        Returns:
            None

        Raises:
            ValueError: if tiles_path has not been specified;
            Exception: if tiles_path is not a directory;
            Exception: if tiles_path doesn't have imagery/labels directories;
            Exception: if imagery/label directories have different numbers of files.
            Exception: if imagery/label tile filenames do not match.
        """

        # check whether tiles_path has been specified
        if self.tiles_path == "":
            raise ValueError("The tiles_path attribute cannot be the empty string.")

        # check whether tiles_path is a directory
        if isdir(self.tiles_path):
            pass
        else:
            raise Exception(self.tiles_path + " is not a directory.")

        # check if imagery/labels subdirectories exist
        tiles_path_contents = listdir(self.tiles_path)
        if 'imagery' in tiles_path_contents and 'labels' in tiles_path_contents:
            pass
        else:
            raise Exception("The tiles_path does not have either imagery or labels subdirectories.")

        # get the image and label tile names
        image_tiles = listdir(join(self.tiles_path, 'imagery'))
        label_tiles = listdir(join(self.tiles_path, 'labels'))

        # check if there are equal numbers of imagery/label tiles
        if len(image_tiles) == len(label_tiles):
            pass
        else:
            raise Exception("The numbers of imagery and label tiles are different.")

        # check that imagery/label tile filenames are the same
        if set(image_tiles) == set(label_tiles):
            pass
        else:
            raise Exception("The imagery and label tiles do not have the same filenames.")

    def generate_tiles(self, stride_length: int = 0,
                       label_proportion: float = 0.2,
                       verbose: bool = True) -> None:

        """Generates image tiles from the source and label imagery for use in model training.

        Args:
            stride_length: the number of pixels of overlap between tiles;
            label_proportion: the minimum proportion which any single class must have per tile;
            verbose: whether to print progress to the console.

        Returns:
            None

        Raises:
            ValueError: if argument stride_length is a negative integer;
            ValueError: if argument label_proportion is not between 0.0 and 1.0, inclusive;
            TypeError: if argument verbose is not boolean.
        """

        # attempt to coerce arguments to the correct type
        stride_length = int(stride_length)
        label_proportion = float(label_proportion)

        # check for correct ranges
        if stride_length < 0:
            raise ValueError('Argument stride_length must be a non-negative integer.')

        if label_proportion < 0.0 or label_proportion > 1.0:
            raise ValueError('Argument label_proportion must be between 0.0 and 1.0, inclusive.')

        if not isinstance(verbose, bool):
            raise TypeError('Argument verbose must be either True or False.')

        # check whether the label rasters are in good shape
        self.check_labels()

        # create sub-directories for the tiles
        imagery_tiles_dir = join(self.tiles_path, 'imagery')
        label_tiles_dir = join(self.tiles_path, 'labels')
        if not (isdir(self.tiles_path)):
            mkdir(self.tiles_path)
        if not isdir(imagery_tiles_dir):
            mkdir(imagery_tiles_dir)
        if not isdir(label_tiles_dir):
            mkdir(label_tiles_dir)

        # loop through each source/label raster pair to generate tiles
        for filename in self.source_image_names:
            # open rgb and raster label imagery
            image = gdal.Open(join(self.source_path, filename))
            labels = gdal.Open(join(self.labels_path, filename))

            # pull out tiles from imagery
            tile_raster_pair(image=image,
                             labels=labels,
                             tile_dimension=self.tile_dimension,
                             stride_length=stride_length,
                             label_proportion=label_proportion,
                             imagery_tiles_dir=imagery_tiles_dir,
                             label_tiles_dir=label_tiles_dir,
                             filename=filename)

            if verbose:
                print(filename + " tiles generated.")

    def get_tile_quantities(self) -> dict:

        """Computes the number of tiles generated from each source image.

        Returns:
            A dictionary containing the number of tiles for each source image."""

        # check that the tiles have been generated properly
        self.check_tiles()

        # define the dictionary to hold tile quantities
        quantity_dict = {}

        # get a list of names of all tiles
        tile_filenames = listdir(join(self.tiles_path, 'imagery'))

        # loop through source image names
        for fname in self.data_names:
            tile_subset = [x for x in tile_filenames if fname in x]
            quantity_dict[fname] = len(tile_subset)

        return quantity_dict

    def rasterize_polygon_layers(self, burn_feature: str = 'bool',
                                 no_data_value: int = 0,
                                 verbose: bool = True) -> None:

        """Generates label rasters from the polygon data, with dimensions matching the source imagery.

        Args:
            burn_feature: the feature from the attribute table to write;
            no_data_value: the value to burn into NoData pixels;
            verbose: whether to print progress to the console.

        Returns:
            None

        Raises:
            TypeError: if verbose argument is not boolean.
        """

        # check for correct argument type
        if not isinstance(verbose, bool):
            raise TypeError('Argument verbose must be either True or False.')

        # check whether polygon data has been set and matches the names in the source imagery
        self.check_polygons()

        # create a label directory if necessary
        if not (isdir(self.labels_path)):
            mkdir(self.labels_path)

        # loop through the shapefiles in the polygons directory
        for filename in self.data_names:
            fname = splitext(filename)[0]
            # open the source/polygon pair
            rgb = gdal.Open(join(self.source_path, filename + '.tif'))
            polygons = ogr.Open(join(self.polygons_path, fname, fname + '.shp'))

            # set the output path
            output_path = join(self.labels_path, filename + '.tif')

            # rasterize the polygon layer

            rasterize_polygon_layer(rgb=rgb,
                                    polygons=polygons,
                                    output_path=output_path,
                                    burn_attribute=burn_feature,
                                    no_data_value=no_data_value)

            if verbose:
                print(filename + " rasterized.")

    def resample_source_imagery(self, output_path: str,
                                target_resolutions: tuple,
                                resample_algorithm: str = 'cubic',
                                replace_source_dataset: bool = True,
                                verbose: bool = True) -> None:

        """Resamples the source imagery to the target resolution.

        Args:
            output_path: the directory to hold the resampled imagery;
            target_resolutions: a tuple of the form (xRes, yRes) for target resolutions, in units of meters;
            resample_algorithm: the method used for resampling (see gdalwarp documentation for more options);
            replace_source_dataset: whether to use the resampled imagery as the new source imagery;
            verbose: whether to print progress to the console.

        Returns:
            None

        Raises:
            TypeError: if argument replace_source_dataset is not boolean;
            TypeError: if argument verbose is not boolean.
        """

        # attempt to coerce arguments to the correct type
        output_path = str(output_path)
        target_resolutions = tuple(target_resolutions)
        resample_algorithm = str(resample_algorithm)

        # check for correct types
        if not isinstance(replace_source_dataset, bool):
            raise TypeError('Argument replace_source_dataset must be either True or False.')

        if not isinstance(verbose, bool):
            raise TypeError('Argument verbose must be either True or False.')

        # create directory if it doesn't already exist
        if not isdir(output_path):
            mkdir(output_path)

        # resample the source rasters
        for filename in self.source_image_names:
            resample_dataset(input_path=join(self.source_path, filename),
                             output_path=join(output_path, filename),
                             resample_algorithm=resample_algorithm,
                             target_resolutions=target_resolutions)

            if verbose:
                print(filename + " resampled to " + str(target_resolutions) + ".")

        # change dataset's source imagery if requested
        if replace_source_dataset:
            self.source_path = output_path

    def set_label_imagery(self, labels_path: str) -> None:

        """Defines the label imagery to use for other methods, if not already created by rasterize_polygons.

        Args:
            labels_path: the directory containing the label rasters.

        Returns:
            None
        """

        # attempt to coerce argument to the correct type
        labels_path = str(labels_path)

        self.labels_path = labels_path
        self.check_labels()

    def set_label_polygons(self, polygons_path: str) -> None:
        
        """Defines the polygon data, and provides a manual alternative to the get_label_polygons method; e.g., this
        method is useful when one wants to use polygon data from a source other than OpenStreetMaps.

        Args:
            polygons_path: the directory containing the polygon data (one sub-directory for each source image).

        Returns:
            None
        """

        # attempt to coerce argument to the correct type
        polygons_path = str(polygons_path)

        self.polygons_path = polygons_path
        self.check_polygons()

    def tf_dataset(self, n_classes: int = 2,
                   batch_size: int = 1,
                   augmentation: bool = True,
                   perform_one_hot: bool = False) -> tf.data.Dataset:

        """Creates a tf.data.Dataset object which generates batches from the tile folders.

        Args:
            n_classes: the number of label classes;
            batch_size: the size of the batch to generate;
            augmentation: whether to perform data augmentation (random flips);
            perform_one_hot: whether to do a one-hot encoding on the label tiles.

        Returns:
            A tf.data.Dataset object.

        Raises:
            ValueError: if argument n_classes is not a positive integer;
            ValueError: if argument batch_size is not a positive integer;
            TypeError: if argument augmentation is not boolean;
            TypeError: if argument perform_one_hot is not boolean.
        """

        # attempt to coerce arguments to the correct type
        n_classes = int(n_classes)
        batch_size = int(batch_size)

        # check for the correct types and ranges
        if n_classes <= 0:
            raise ValueError('Argument n_classes must be a positive integer.')

        if batch_size <= 0:
            raise ValueError('Argument batch_size must be a positive integer.')

        if not isinstance(augmentation, bool):
            raise TypeError('Argument augmentation must be either True or False.')

        if not isinstance(perform_one_hot, bool):
            raise TypeError('Argument perform_one_hot must be either True or False.')

        @tf.function
        def augment(img, lbl):
            # perform a random vertical flip
            img, lbl = tf.cond(tf.greater(tf.random.uniform(()), 0.5),
                               lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(lbl)),
                               lambda: (img, lbl))

            # perform a random rotation
            rot_val = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
            img = tf.image.rot90(img, k=rot_val)
            lbl = tf.image.rot90(lbl, k=rot_val)

            return img, lbl

        # define a generator object which will then define a tf.data.Dataset
        def generator():
            filenames = listdir(join(self.tiles_path, 'imagery'))
            if augment:
                shuffle(filenames)
            train_ids = range(len(filenames))
            while True:
                for ID in train_ids:
                    img = gdal.Open(join(self.tiles_path, 'imagery', filenames[ID])).ReadAsArray()
                    lbl = gdal.Open(join(self.tiles_path, 'labels', filenames[ID])).ReadAsArray()

                    # reshape img to channels-last
                    img = moveaxis(img, 0, -1)

                    # perform a one-hot encoding of the labels (creates n_channels dimension either way)
                    if perform_one_hot:
                        lbl = convert_labels_to_one_hots(lbl, n_classes)
                    else:
                        lbl = expand_dims(lbl, -1)

                    yield img, lbl

        # create the tf.data.Dataset object
        if perform_one_hot:
            output_signature = (
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, self.n_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, n_classes), dtype=tf.float32))
        else:
            output_signature = (
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, self.n_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, 1), dtype=tf.float32))

        tf_ds = tf.data.Dataset.from_generator(
            generator,
            args=[],
            output_signature=output_signature)

        # define the tf.data.Dataset
        if augmentation:
            tf_ds = (
                tf_ds
                .shuffle(10 * batch_size, reshuffle_each_iteration=True)
                .batch(batch_size)
                .map(lambda x, y: augment(x, y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            tf_ds = (
                tf_ds
                .shuffle(10 * batch_size, reshuffle_each_iteration=True)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        return tf_ds
