# datasets.py

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
        self.source_images: list = []
        self.source_path: str = ""
        self.source_extent: float = 0.0
        self.source_pixel_units: str = ""
        self.source_resolutions: dict = {}
        self.tile_dimension: int = 0
        self.tile_path: str = ""

    def generate_tiles(self, dimension: int,
                       tile_path: str,
                       drop_single_class_tiles: bool = True) -> None:

        raise NotImplementedError("Method \'generate_tiles\' not implemented.")

    def get_label_polygons(self, osm_keys: list) -> None:

        raise NotImplementedError("Method \'get_label_polygons\' not implemented.")

    def rasterize_labels(self, burn_value: int,
                         no_data_value: int) -> None:

        raise NotImplementedError("Method \'rasterize_labels\' not implemented.")

    def resample(self, method: str,
                 target_resolution: float) -> None:

        raise NotImplementedError("Method \'resample\' not implemented.")

    def return_batch(self):

        raise NotImplementedError("Method \'return_batch\' not implemented.")

    def set_label_imagery(self, rasters_path: str) -> None:

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_label_polygons(self, polygons_path) -> None:

        raise NotImplementedError("Method \'set_labels\' not implemented.")

    def set_source_imagery(self, path: str) -> None:

        raise NotImplementedError("Method \'set_source_imagery\' not implemented.")
