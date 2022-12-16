# datasets.py

from geode import utilities
import numpy as np
import os
from osgeo import gdal, ogr



class SemanticSegmentation:
    """Defines a semantic segmentation dataset to be used in deep-learning models. Has methods to get polygon layers
     from OpenStreetMaps, to rasterize polygon layers, to generate training tiles, and to generate an iterator object
     for model training."""

    def __init__(self, source_path: str = "",
                 vector_path: str = "",
                 raster_path: str = "",
                 tile_dimension: int = 0,
                 tile_path: str = "",
                 dataset_description: str = "",
                 channel_description: str = "",
                 no_data_value: int = 0,
                 burn_value: int = 1,
                 osm_key: str = "building"):

        self.channel_description: str = channel_description
        self.dataset_description: str = dataset_description
        self.raster_path = raster_path
        self.source_image_names = os.listdir(source_path)
        self.source_metadata: dict = {}
        self.source_path = source_path
        self.vector_path = vector_path
        self.tile_dimension = tile_dimension
        self.tile_path = tile_path
        self.data_names = [os.path.splitext(x)[0] for x in os.listdir(source_path)]
        self.no_data_value = no_data_value
        self.burn_value = burn_value

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
        elif len(os.listdir(self.source_path)) == 0:
            raise Exception("The source_path is empty.")

    def check_vectors(self) -> None:
        """Checks whether the vector data has been set, and matches the names of the source imagery.
        
        Returns:
            None

        Raises:
            Exception: if vector_path has not been set;
            Exception: if vector_path is empty;
            Exception: if vector data folder names do not match source imagery names;
            Exception: if a vector data folder does not contain a shapefile.
        """

        if self.vector_path == "":
            raise Exception("The vector_path has not been set; run either the get_label_vectors "
                            "or set_label_vectors first.")
        elif len(os.listdir(self.vector_path)) == 0:
            raise Exception("The vector_path is empty.")
        elif [os.path.splitext(x)[0] for x in os.listdir(self.source_path)] != os.listdir(self.vector_path):
            raise Exception("Source imagery names do not match vector data names.")
        else:
            for directory in os.listdir(self.vector_path):
                filenames = os.listdir(os.path.join(self.vector_path, directory))
                shapefiles = [x for x in filenames if os.path.splitext(x)[1] == ".shp"]
                if len(shapefiles) != 1:
                    raise Exception("The vector data directories must have exactly one shapefile.")

    def check_rasters(self) -> None:
        """Checks whether the label rasters have been generated, and match the names of the source imagery.

        Returns:
            None

        Raises:
            Exception: if raster_path has not been set;
            Exception: if raster_path is empty;
            Exception: if label raster names do not match source imagery names;
            Exception: if source/raster width does not match for a particular pair;
            Exception: if source/raster height does not match for a particular pair;
            Exception: if source/raster projections do not match for a particular pair.
        """

        if self.raster_path == "":
            raise Exception("The raster_path has not been set; run either the set_label_imagery "
                            "or rasterize_vectors first.")
        elif len(os.listdir(self.raster_path)) == 0:
            raise Exception("The raster_path is empty.")
        elif os.listdir(self.source_path) != os.listdir(self.raster_path):
            raise Exception("Source imagery names do not match label raster names.")
        else:
            for filename in self.source_image_names:
                source_dataset = gdal.Open(os.path.join(self.source_path, filename))
                label_dataset = gdal.Open(os.path.join(self.raster_path, filename))

                # we use a numpy unittest to determine if the geotransforms are almost equal:
                np.testing.assert_allclose(source_dataset.GetGeoTransform(), label_dataset.GetGeoTransform())

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
            Exception: if tile_path has not been specified;
            Exception: if tile_path is not a directory;
            Exception: if tile_path doesn't have imagery/labels directories;
            Exception: if imagery/label directories have different numbers of files.
            Exception: if imagery/label tile filenames do not match.
        """

        # check whether tile_path has been specified
        if self.tile_path == "":
            raise Exception("The tile_path attribute has not been specified.")

        # check whether tile_path is a directory
        if os.path.isdir(self.tile_path):
            pass
        else:
            raise Exception(self.tile_path + " is not a directory.")

        # check if imagery/labels subdirectories exist
        tile_path_contents = os.listdir(self.tile_path)
        if "imagery" in tile_path_contents and "labels" in tile_path_contents:
            pass
        else:
            raise Exception("The tile_path does not have either imagery or labels subdirectories.")

        # get the image and label tile names
        image_tiles = os.listdir(os.path.join(self.tile_path, "imagery"))
        label_tiles = os.listdir(os.path.join(self.tile_path, "labels"))

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

    def get_source_metadata(self) -> None:
        """Compiles information about the source imagery, and stores it in the source_metadata attribute.

        Returns:
            None
        """

        # check whether the source imagery exists
        self.check_source()

        # loop through the source imagery and extract relevant metadata
        for filename in self.source_image_names:
            dst = gdal.Open(os.path.join(self.source_path, filename))

            # set metadata fields from the gdal.Dataset object
            metadata_dict = {}
            gt = dst.GetGeoTransform()
            x_dim = dst.RasterXSize
            y_dim = dst.RasterYSize

            metadata_dict["band_counts"] = dst.RasterCount
            metadata_dict["dimensions"] = (x_dim, y_dim)
            metadata_dict["resolution"] = (gt[1], gt[5])
            metadata_dict["raster_extent"] = np.abs(x_dim * gt[1] * y_dim * gt[5])

            # calculate the "data_extent", or the raster_extent minus the area of nodata pixels. First, compute the
            # number of nodata pixels
            n_nodata_pixels = np.sum(np.where(np.sum(dst.ReadAsArray()) == 0, 1, 0))

            # then compute the area of these pixels
            nodata_area = np.abs(n_nodata_pixels * x_dim * y_dim)

            # set the data_extent in the metadata_dictionary
            metadata_dict["data_extent"] = metadata_dict["raster_extent"] - nodata_area

            if "metre" in dst.GetProjection():
                metadata_dict["units"] = "metre"
            else:
                raise (Exception("The raster " + filename + " may not be in metres."))

            self.source_metadata[filename] = metadata_dict

            # close the gdal.Dataset object
            dst = None

    def generate_tiles(self, drop_single_class_tiles: bool = True,
                       verbose: bool = True) -> None:
        """Generates image tiles from the source and label imagery for use in model training. I followed the process
        given in the video https://www.youtube.com/watch?v=H5uQ85VXttg.

        Args:
            drop_single_class_tiles: whether to ignore tiles with a single class;
            verbose: whether to print progress to the console.

        Returns:
            None
        """

        # check whether the label rasters are in good shape
        self.check_rasters()

        # create sub-directories for the tiles
        imagery_tiles_dir = os.path.join(self.tile_path, "imagery")
        label_tiles_dir = os.path.join(self.tile_path, "labels")
        if not (os.path.isdir(self.tile_path)):
            os.mkdir(self.tile_path)
        if not os.path.isdir(imagery_tiles_dir):
            os.mkdir(imagery_tiles_dir)
        if not os.path.isdir(label_tiles_dir):
            os.mkdir(label_tiles_dir)

        # loop through each source/label raster pair to generate tiles
        for filename in self.source_image_names:
            # open rgb and raster label imagery
            rgb = gdal.Open(os.path.join(self.source_path, filename))
            labels = gdal.Open(os.path.join(self.raster_path, filename))

            # pull out tiles from imagery
            utilities.tile_raster_pair(rgb=rgb,
                                   labels=labels,
                                   tile_dimension=self.tile_dimension,
                                   drop_single_class_tiles=drop_single_class_tiles,
                                   imagery_tiles_dir=imagery_tiles_dir,
                                   label_tiles_dir=label_tiles_dir,
                                   filename=filename)

            if verbose:
                print(filename + " tiles generated.")

    def get_label_vectors(self) -> None:
        """Queries the OpenStreetMaps API and downloads vector data over the source imagery.

        Returns:
            None
        """

        # check whether the source imagery directory has been set and is nonempty.
        self.check_vectors()

        # loop through the source files
        # for filename in self.data_names:

        # raise NotImplementedError("Method \'get_label_polygons\' not implemented.")

    def rasterize_polygon_layers(self, verbose=True) -> None:
        """Generates label rasters from the vector data, with dimensions matching the source imagery.

        Args:
            verbose: whether to print progress to the console.

        Returns:
            None
        """
        
        # check whether vector data has been set and matches the names in the source imagery
        self.check_vectors()

        # loop through the shapefiles in the vectors directory
        for filename in self.data_names:
            fname = os.path.splitext(filename)[0]
            # open the source/polygon pair
            rgb = gdal.Open(os.path.join(self.source_path, filename + ".tif"))
            polygons = ogr.Open(os.path.join(self.vector_path, fname, fname + ".shp"))

            # set the output path
            output_path = os.path.join(self.raster_path, filename + ".tif")

            # rasterize the polygon layer

            utilities.rasterize_polygon_layer(rgb=rgb,
                                          polygons=polygons,
                                          output_path=output_path,
                                          burn_value=self.burn_value,
                                          no_data_value=self.no_data_value)

            if verbose:
                print(filename + " rasterized.")

    def resample_source_imagery(self, output_path: str,
                                target_resolution: tuple,
                                resample_algorithm: str="cubic",
                                replace_source_dataset: bool = True,
                                verbose=True) -> None:
        """Resamples the source imagery to the target resolution.

        Args:
            output_path: the directory to hold the resampled imagery;
            target_resolution: a tuple of the form (xRes, yRes) for target resolutions, in units of meters;
            resample_algorithm: the method used for resampling (see gdalwarp documentation for more options);
            replace_source_dataset: whether to use the resampled imagery as the new source imagery;
            verbose: whether to print progress to the console.

        Returns:
            None
        """

        # create directory if it doesn't already exist
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        # resample the rasters
        for filename in self.source_image_names:
            utilities.resample_dataset(input_path=os.path.join(self.source_path, filename),
                                   output_path=os.path.join(output_path, filename),
                                   resample_algorithm=resample_algorithm,
                                   target_resolution=target_resolution)

            if verbose:
                print(filename + " resampled to " + str(target_resolution) + ".")

        # change dataset's source imagery if requested
        if replace_source_dataset:
            self.source_path = output_path

    def set_label_imagery(self, raster_path: str) -> None:
        """Defines the label imagery to use for other methods, if not already created by rasterize_vectors.

        Args:
            raster_path: the directory containing the label rasters.

        Returns:
            None
        """

        self.raster_path = raster_path
        self.check_rasters()

    def set_label_vectors(self, vector_path) -> None:
        """Defines the vector data, and provides a manual alternative to the get_label_vectors method; e.g., this method
        is useful when one wants to use vector data from a source other than OpenStreetMaps.

        Args:
            vector_path: the directory containing the vector data (one sub-directory for each source image).

        Returns:
            None
        """

        self.vector_path = vector_path
        self.check_vectors()

    def training_generator(self, batch_size: int,
                           use_tiles: bool=True,
                           perform_one_hot: bool=True,
                           n_classes: int=2,
                           flip_vertically: bool=True,
                           rotate: bool=True,
                           scale_factor: float=1/255) -> iter:
        """Creates an iterator object for model training.

        Args:
            batch_size: the number of tile pairs in each batch;
            use_tiles: if true, uses the files at tile_path; otherwise, it reads tiles from the source/label pairs;
            perform_one_hot: whether to do a one-hot encoding on the label tiles;
            n_classes: the number of label classes;
            flip_vertically: whether to randomly flip tile pairs vertically;
            rotate: whether to randomly rotate tile pairs;
            scale_factor: the factor by which to rescale input imagery tiles.

        Returns:
            An iterator object which generates batches of tile pairs for training.

        Raises:
            Exception: if perform_one_hot is False and n_classes is not an integer greater than 1.
        """

        if use_tiles:
            # check that tiles are correctly generated
            self.check_tiles()

            if perform_one_hot and n_classes < 2:
                raise Exception("Number of classes must be larger than two when performing one-hot encoding.")

            imagery_path = os.path.join(self.tile_path, "imagery")
            labels_path = os.path.join(self.tile_path, "labels")

            imagery_filenames = os.listdir(imagery_path)
            np.random.shuffle(imagery_filenames)
            ids_train_split = range(len(imagery_filenames))
            while True:
                for start in range(0, len(ids_train_split), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(ids_train_split))
                    ids_train_batch = ids_train_split[start:end]

                    for ID in ids_train_batch:
                        img = gdal.Open(os.path.join(imagery_path, imagery_filenames[ID])).ReadAsArray()
                        lbl = gdal.Open(os.path.join(labels_path, imagery_filenames[ID])).ReadAsArray()

                        # ensure tiles follow the channels-last convention
                        img = np.moveaxis(img, 0, -1)

                        # perform a random counterclockwise rotation
                        if rotate:
                            k_rot = np.random.randint(0, 4)
                            img = np.rot90(img, k=k_rot)
                            lbl = np.rot90(lbl, k=k_rot)

                        # perform a random vertical flip
                        if flip_vertically and np.random.randint(0, 2) == 1:
                            img = np.flip(img, axis=0)
                            lbl = np.flip(lbl, axis=0)

                        # rescale the imagery tile
                        img = img * scale_factor

                        x_batch.append(img)
                        y_batch.append(lbl)

                    x_batch = np.array(x_batch)
                    y_batch = np.array(y_batch)

                    # perform a one-hot encoding if desired
                    if perform_one_hot:
                        oh_y_batch = np.zeros(y_batch.shape + (n_classes, ), dtype=np.uint8)
                        for i in range(n_classes):
                            oh_y_batch[:, :, :, i][y_batch == i] = 1

                        y_batch = oh_y_batch

                    yield x_batch, y_batch
        else:
            pass
