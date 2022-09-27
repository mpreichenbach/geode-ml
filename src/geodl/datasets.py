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
        self.raster_labels_path: str = ""
        self.resampled_resolutions: dict = {}
        self.source_image_names: list = []
        self.source_path: str = ""
        self.source_extents: dict = {}
        self.source_pixel_units: str = ""
        self.source_resolutions: dict = {}
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

        raise NotImplementedError("Method \'set_source_imagery\' not implemented.")

