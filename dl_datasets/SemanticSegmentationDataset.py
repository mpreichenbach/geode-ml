# SemanticSegmentationDataset.py

from osgeo import gdal
from osgeo import ogr


class SemanticSegmentationDataset:
    """
    Defines a semantic segmentation dataset to be used in deep-learning models.
    """
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

    def generate_tiles(self, dimension: int,
                       tile_path: str,
                       class_proportions: None,
                       drop_single_class_tiles: bool = True,
                       n_images: int = 0) -> None:

        raise NotImplementedError("Method \'generate_tiles\' not implemented.")

    def get_labels(self, osm_classes: list) -> None:

        raise NotImplementedError("Method \'get_labels\' not implemented.")

    def load_source_image(self, value: int) -> gdal.Dataset:

        raise NotImplementedError("Method \'load_source_image\' not implemented.")

    def set_source_imagery(self, path: str) -> None:

        raise NotImplementedError("Method \'get_source_imagery\' not implemented.")

    def rasterize_labels(self, burn_value: int,
                         no_data_value: int) -> None:

        raise NotImplementedError("Method \'rasterize_labels\' not implemented.")

    def resample(self, method: str,
                 target_resolution: float) -> None:

        raise NotImplementedError("Method \'resample\' not implemented.")

    def return_batch(self):

        raise NotImplementedError("Method \'return_batch\' not implemented.")

    def save_dataset(self, path: str) -> bool:

        raise NotImplementedError("Method \'save_dataset\' not implementd.")

    def set_labels(self, path: str) -> None:

        raise NotImplementedError("Method \'set_labels\' not implemented.")
