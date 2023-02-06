# datasets.py

from geode.utilities import convert_labels_to_one_hots, rasterize_polygon_layer, resample_dataset, tile_raster_pair
from numpy import abs, flip, moveaxis, rot90
from numpy.testing import assert_allclose
from numpy.random import randint, shuffle
from os import listdir, mkdir
from os.path import isdir, join, splitext
from osgeo import gdal, ogr
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential


class SegmentationDataset:
    """Defines a semantic segmentation dataset to be used in deep-learning models. Has methods to resample source
     imagery, to rasterize polygon layers, to generate training tiles, and to generate an iterator object
     for model training."""

    def __init__(self, source_path: str = "",
                 polygons_path: str = "",
                 labels_path: str = "",
                 n_channels: int = 3,
                 tile_dimension: int = 0,
                 tiles_path: str = "",
                 dataset_description: str = "",
                 channel_description: str = "",
                 no_data_value: int = 0,
                 burn_attribute: str = "bool"):

        self.n_channels = n_channels
        self.channel_description: str = channel_description
        self.dataset_description: str = dataset_description
        self.label_proportion: float = 0.0
        self.labels_path = labels_path
        self.source_metadata: dict = {}
        self.source_path = source_path
        self.polygons_path = polygons_path
        self.tile_dimension = tile_dimension
        self.tiles_path = tiles_path
        self.no_data_value = no_data_value
        self.burn_attribute = burn_attribute

        if source_path == "":
            pass
        else:
            self.source_image_names = listdir(source_path)
            self.data_names = [splitext(x)[0] for x in self.source_image_names]

    def check_source(self) -> None:
        """Checks whether the source imagery has been set, and is nonempty.

        Returns:
            None

        Raises:
            Exception: if source_path has not been set;
            Exception: if source_path has been set, but is empty.
        """

        if self.source_path == "":
            raise Exception("The source_path has not been set.")
        elif len(listdir(self.source_path)) == 0:
            raise Exception("The source_path is empty.")

    def check_polygons(self) -> None:
        """Checks whether the polygon data has been set, and matches the names of the source imagery.

        Returns:
            None

        Raises:
            Exception: if polygons_path has not been set;
            Exception: if polygons_path is empty;
            Exception: if polygon data folder names do not match source imagery names;
            Exception: if a polygon data folder does not contain a shapefile.
        """

        if self.polygons_path == "":
            raise Exception("The polygons_path has not been set; run either the get_label_polygons "
                            "or set_label_polygons first.")
        elif len(listdir(self.polygons_path)) == 0:
            raise Exception("The polygons_path is empty.")
        elif set([splitext(x)[0] for x in listdir(self.source_path)]) != set(listdir(self.polygons_path)):
            raise Exception("Source imagery names do not match polygon data names.")
        else:
            for directory in listdir(self.polygons_path):
                filenames = listdir(join(self.polygons_path, directory))
                shapefiles = [x for x in filenames if splitext(x)[1] == ".shp"]
                if len(shapefiles) != 1:
                    raise Exception("The polygons data directories must have exactly one shapefile.")

    def check_labels(self) -> None:
        """Checks whether the label rasters have been generated, and match the names of the source imagery.

        Returns:
            None

        Raises:
            Exception: if labels_path has not been set;
            Exception: if labels_path is empty;
            Exception: if label raster names do not match source imagery names;
            Exception: if source/raster width does not match for a particular pair;
            Exception: if source/raster height does not match for a particular pair;
            Exception: if source/raster projections do not match for a particular pair.
        """

        if self.labels_path == "":
            raise Exception("The labels_path has not been set; run either the set_label_imagery "
                            "or rasterize_polygons first.")
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
            Exception: if tiles_path has not been specified;
            Exception: if tiles_path is not a directory;
            Exception: if tiles_path doesn't have imagery/labels directories;
            Exception: if imagery/label directories have different numbers of files.
            Exception: if imagery/label tile filenames do not match.
        """

        # check whether tiles_path has been specified
        if self.tiles_path == "":
            raise Exception("The tiles_path attribute has not been specified.")

        # check whether tiles_path is a directory
        if isdir(self.tiles_path):
            pass
        else:
            raise Exception(self.tiles_path + " is not a directory.")

        # check if imagery/labels subdirectories exist
        tiles_path_contents = listdir(self.tiles_path)
        if "imagery" in tiles_path_contents and "labels" in tiles_path_contents:
            pass
        else:
            raise Exception("The tiles_path does not have either imagery or labels subdirectories.")

        # get the image and label tile names
        image_tiles = listdir(join(self.tiles_path, "imagery"))
        label_tiles = listdir(join(self.tiles_path, "labels"))

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

    def get_source_metadata(self) -> dict:
        """Compiles information about the source imagery, and stores it in the source_metadata attribute.

        Returns:
            None
        """

        # check whether the source imagery exists
        self.check_source()

        # loop through the source imagery and extract relevant metadata
        for filename in self.source_image_names:
            dst = gdal.Open(join(self.source_path, filename))

            # set metadata fields from the gdal.Dataset object
            metadata_dict = {}
            gt = dst.GetGeoTransform()
            x_dim = dst.RasterXSize
            y_dim = dst.RasterYSize

            metadata_dict["band_counts"] = dst.RasterCount
            metadata_dict["dimensions"] = (x_dim, y_dim)
            metadata_dict["resolution"] = (gt[1], gt[5])
            metadata_dict["raster_extent"] = abs(x_dim * gt[1] * y_dim * gt[5])

            if "metre" in dst.GetProjection():
                metadata_dict["units"] = "metre"
            else:
                raise (Exception("The raster " + filename + " may not be in metres."))

            self.source_metadata[filename] = metadata_dict

            # close the gdal.Dataset object
            dst = None

        return self.source_metadata

    def generate_tiles(self, label_proportion: float = 0.2,
                       verbose: bool = True) -> None:
        """Generates image tiles from the source and label imagery for use in model training. I followed the process
        given in the video https://www.youtube.com/watch?v=H5uQ85VXttg.

        Args:
            label_proportion: the minimum proportion which any single class must have per tile;
            verbose: whether to print progress to the console.

        Returns:
            None
        """

        # check whether the label rasters are in good shape
        self.check_labels()

        # set an attribute for the label proportion in the generated tiles
        self.label_proportion = label_proportion

        # create sub-directories for the tiles
        imagery_tiles_dir = join(self.tiles_path, "imagery")
        label_tiles_dir = join(self.tiles_path, "labels")
        if not (isdir(self.tiles_path)):
            mkdir(self.tiles_path)
        if not isdir(imagery_tiles_dir):
            mkdir(imagery_tiles_dir)
        if not isdir(label_tiles_dir):
            mkdir(label_tiles_dir)

        # loop through each source/label raster pair to generate tiles
        for filename in self.source_image_names:
            # open rgb and raster label imagery
            rgb = gdal.Open(join(self.source_path, filename))
            labels = gdal.Open(join(self.labels_path, filename))

            # pull out tiles from imagery
            tile_raster_pair(rgb=rgb,
                             labels=labels,
                             tile_dimension=self.tile_dimension,
                             label_proportion=self.label_proportion,
                             imagery_tiles_dir=imagery_tiles_dir,
                             label_tiles_dir=label_tiles_dir,
                             filename=filename)

            if verbose:
                print(filename + " tiles generated.")

    def rasterize_polygon_layers(self, verbose=True) -> None:
        """Generates label rasters from the polygon data, with dimensions matching the source imagery.

        Args:
            verbose: whether to print progress to the console.

        Returns:
            None
        """

        # check whether polygon data has been set and matches the names in the source imagery
        self.check_polygons()

        # loop through the shapefiles in the polygons directory
        for filename in self.data_names:
            fname = splitext(filename)[0]
            # open the source/polygon pair
            rgb = gdal.Open(join(self.source_path, filename + ".tif"))
            polygons = ogr.Open(join(self.polygons_path, fname, fname + ".shp"))

            # set the output path
            output_path = join(self.labels_path, filename + ".tif")

            # rasterize the polygon layer

            rasterize_polygon_layer(rgb=rgb,
                                    polygons=polygons,
                                    output_path=output_path,
                                    burn_attribute=self.burn_attribute,
                                    no_data_value=self.no_data_value)

            if verbose:
                print(filename + " rasterized.")

    def resample_source_imagery(self, output_path: str,
                                target_resolutions: tuple,
                                resample_algorithm: str = "cubic",
                                replace_source_dataset: bool = True,
                                verbose=True) -> None:
        """Resamples the source imagery to the target resolution.

        Args:
            output_path: the directory to hold the resampled imagery;
            target_resolutions: a tuple of the form (xRes, yRes) for target resolutions, in units of meters;
            resample_algorithm: the method used for resampling (see gdalwarp documentation for more options);
            replace_source_dataset: whether to use the resampled imagery as the new source imagery;
            verbose: whether to print progress to the console.

        Returns:
            None
        """

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
            self.get_source_metadata()

    def set_label_imagery(self, labels_path: str) -> None:
        """Defines the label imagery to use for other methods, if not already created by rasterize_polygons.

        Args:
            labels_path: the directory containing the label rasters.

        Returns:
            None
        """

        self.labels_path = labels_path
        self.check_labels()

    def set_label_polygons(self, polygons_path) -> None:
        """Defines the polygon data, and provides a manual alternative to the get_label_polygons method; e.g., this
        method is useful when one wants to use polygon data from a source other than OpenStreetMaps.

        Args:
            polygons_path: the directory containing the polygon data (one sub-directory for each source image).

        Returns:
            None
        """

        self.polygons_path = polygons_path
        self.check_polygons()

    def tf_dataset(self, n_classes: int = 2,
                   augment = True,
                   batch_size: int = 1,
                   perform_one_hot: bool = False) -> tf.data.Dataset:

        """Creates a tf.data.Dataset object which generates batches from the tile folders.

        Args:
            n_classes: the number of label classes;
            flip_vertically: whether to randomly flip tile pairs vertically;
            rotate: whether to randomly rotate tile pairs;
            scale_factor: the factor by which to rescale input imagery tiles;
            perform_one_hot: whether to do a one-hot encoding on the label tiles.

        Returns:
            A tf.data.Dataset object.

        Raises:
            Exception: if tile_dimension attribute has not been set.
        """

        # check that tile_dimension attribute has been set
        if self.tile_dimension == 0:
            raise Exception("The tile_generator attribute must be greater than 0.")

        # first, define a generator object which will define a tf.data.Dataset
        def generator():
            filenames = listdir(join(self.tiles_path, "imagery"))
            train_ids = range(len(filenames))
            while True:
                shuffle(filenames)
                for ID in train_ids:
                    img = gdal.Open(join(self.tiles_path, "imagery", filenames[ID])).ReadAsArray()
                    lbl = gdal.Open(join(self.tiles_path, "labels", filenames[ID])).ReadAsArray()

                    # reshape img to channels-last
                    img = moveaxis(img, 0, -1)

                    # perform a one-hot encoding of the labels
                    if perform_one_hot:
                        lbl = convert_labels_to_one_hots(lbl, n_classes)

                    yield img, lbl

        # create the tf.data.Dataset object
        if perform_one_hot:
            output_signature = (
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, self.n_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, n_classes), dtype=tf.float32))
        else:
            output_signature = (
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension, self.n_channels), dtype=tf.float32),
                tf.TensorSpec(shape=(self.tile_dimension, self.tile_dimension), dtype=tf.float32))

        tf_ds = tf.data.Dataset.from_generator(
            generator,
            args=[],
            output_signature=output_signature)

        # do data augmentation
        if augment:
            ds_aug = Sequential([preprocessing.Rescaling(1.0 / 255),
                                 preprocessing.RandomFlip('horizontal_and_vertical')])

            tf_ds = (
                tf_ds
                .shuffle(2 * batch_size)
                .batch(batch_size)
                .map(lambda x, y: (ds_aug(x), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            tf_ds = (
                tf_ds
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        return tf_ds
