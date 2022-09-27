# datasets.py

import numpy as np
import os
from osgeo import gdal


class SemSeg:
    """Defines a semantic segmentation dataset to be used in deep-learning models."""

    def __init__(self,
                 dataset_description: str = "",
                 channel_description: str = ""):

        self.channel_description: str = channel_description
        self.dataset_description: str = dataset_description
        self.labels: dict = {}
        self.labels_path: str = ""
        self.source_metadata: dict = {}
        self.raster_labels_path: str = ""
        self.resampled_resolutions: dict = {}
        self.source_image_names: list = []
        self.source_path: str = ""
        self.tile_dimension: int = 0
        self.tile_path: str = ""

    def generate_tiles(self, dimension: int,
                       tile_path: str,
                       drop_single_class_tiles: bool = True) -> None:

        raise NotImplementedError("Method \'generate_tiles\' not implemented.")

    def get_label_vectors(self, save_path: str,
                          osm_keys: list) -> None:

        raise NotImplementedError("Method \'get_label_polygons\' not implemented.")

    def rasterize_vectors(self, out_path: str,
                          burn_value: int = 1,
                          no_data_value: int = 0) -> None:

        raise NotImplementedError("Method \'rasterize_vectors\' not implemented.")

    def set_label_imagery(self, raster_path: str) -> None:

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_label_vectors(self, vector_path) -> None:

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_source_imagery(self, source_path: str) -> None:
        """Defines the source imagery to use for other methods; should be run before any other methods."""

        # update source_path attribute
        self.source_path = source_path

        # get imagery names
        self.source_image_names = os.listdir(self.source_path)

        # loop through the imagery and extract relevant metadata
        for filename in self.source_image_names:
            with gdal.Open(os.path.join(self.source_path, filename)) as dst:
                metadata_dict = {}
                metadata_dict["band_counts"] = dst.RasterCount
                metadata_dict["dimensions"] = [dst.RasterXSize, dst.RasterYSize]
                metadata_dict["resolution"] = (dst.GetGeoTransform[1], dst.GetGeoTransform[5])
                metadata_dict["raster_extent"] = abs(dst.RasterXSize * dst.GetGeoTransform[1] *
                                           dst.RasterYSize * dst.GetGeoTransform[5])

                # calculate the "data_extent", or the raster_extent minus the area of nodata pixels. First, compute the
                # number of nodata pixels
                n_nodata_pixels = np.sum(np.where(np.sum(dst.ReadAsArray()) == 0, 1, 0))

                if "metres" in dst.GetGeoTransform():
                    metadata_dict["units"] = "meters"
                else:
                    raise(Exception("The raster " + filename + " may not be in meters."))

                self.source_metadata[filename] = metadata_dict


        raise NotImplementedError("Method \'set_source_imagery\' not implemented.")

