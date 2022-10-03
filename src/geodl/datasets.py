# datasets.py

import numpy as np
import os
from osgeo import gdal
from pathlib import Path


class SemSeg:
    """Defines a semantic segmentation dataset to be used in deep-learning models."""

    def __init__(self,
                 source_path: str = "",
                 vector_path: str = "",
                 raster_path: str = "",
                 tile_dimension: int = 0,
                 tile_path: str = "",
                 dataset_description: str = "",
                 channel_description: str = ""):

        self.channel_description: str = channel_description
        self.dataset_description: str = dataset_description
        self.raster__path = raster_path
        self.resampled_metadata: dict = {}
        self.source_image_names = os.listdir(source_path)
        self.source_metadata: dict = {}
        self.source_path = source_path
        self.vector_path = vector_path
        self.tile_dimension = tile_dimension
        self.tile_path = tile_path

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
                raise(Exception("The raster " + filename + " may not be in metres."))

            self.source_metadata[filename] = metadata_dict

            # close the gdal.Dataset object
            dst = None

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
                            "or set_label_vectors first")
        elif len(os.listdir(self.vector_path)) == 0:
            raise Exception("The vector_path is empty.")
        elif [Path(x).stem for x in os.listdir(self.source_path)] != os.listdir(self.vector_path):
            raise Exception("Source imagery names do not match vector data names.")
        else:
            for directory in os.listdir(self.source_path):
                filenames = os.listdir(directory)
                shapefiles = [x for x in filenames if "shp" in x]
                if len(shapefiles) != 1:
                    raise Exception("A vector data directories must have exactly one shapefile.")

    def generate_tiles(self, dimension: int,
                       tile_path: str,
                       drop_single_class_tiles: bool = True) -> None:
        """Generates image tiles from the source and label imagery for use in model training.

        Args:
            dimension: the side-length in pixels of the square tiles;
            tile_path: the directory in which to save the tiles;
            drop_single_class_tiles: whether to drop tile pairs in which the labels have only a single value.

        Returns:
            None
        """

        # check whether the vector imagery is in good shape
        self.check_vectors()

        raise NotImplementedError("Method \'generate_tiles\' not implemented.")

    def get_label_vectors(self, save_path: str,
                          osm_keys: list) -> None:
        """Queries the OpenStreetMaps API and downloads vector data over the source imagery.

        Args:
            save_path: the directory in which to save the vector data;
            osm_keys: a list of OSM keys describing the kind of data to query.

        Returns:
            None
        """

        # check whether the source imagery directory has been set and is nonempty.
        self.check_vectors()

        raise NotImplementedError("Method \'get_label_polygons\' not implemented.")

    def rasterize_vectors(self, out_path: str,
                          burn_value: int = 1,
                          no_data_value: int = 0) -> None:
        """Generates label rasters from the vector data, with dimensions matching the source imagery.

        Args:
            out_path: the directory in which to save the label rasters;
            burn_value: the value to assign the positive instances of the land-cover type;
            no_data_value: the value to assign the negative instances of the land-cover type.

        Returns:
            None
        """
        
        # check whether vector data has been set and matches the names in the source imagery
        self.check_vectors()

        raise NotImplementedError("Method \'rasterize_vectors\' not implemented.")

    def set_label_imagery(self, raster_path: str) -> None:
        """Defines the label imagery to use for other methods, if not already created by rasterize_vectors.

        Args:
            raster_path: the directory containing the label rasters.

        Returns:
            None
        """

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_label_vectors(self, vector_path) -> None:
        """Defines the vector data, and provides a manual alternative to the get_label_vectors method; e.g., this method
        is useful when one wants to use vector data from a source other than OpenStreetMaps.

        Args:
            vector_path: the directory containing the vector data (one sub-directory for each source image).

        Returns:
            None
        """

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_source_imagery(self, source_path: str) -> None:
        """Defines the source imagery for the dataset; should be run before any other methods.

        Args:
            source_path: the directory containing tifs of the source imagery.

        Returns:
            None
        """

