# test_set_source_imagery.py

import numpy as np
import os
from osgeo import gdal, ogr
import shutil
from src.geodl.datasets import SemSeg
import unittest


class BaseTestGeodl(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""

        cls.test_channel_description = "RGB"
        cls.test_dataset_description = "An image over Bellingham, WA."
        cls.tile_dimension = 512
        cls.test_image_path = "test/imagery/source/"
        cls.test_osm_keys = ["building"]
        cls.test_osm_vector_path = "test/imagery/osm_label_vectors"
        cls.test_vector_path = "test/imagery/label_vectors"
        cls.test_raster_path = "test/imagery/label_rasters"
        cls.test_tile_path = "test/imagery/tiles"

        cls.dataset = SemSeg(dataset_description=cls.test_dataset_description,
                             channel_description=cls.test_channel_description)

        cls.n_source_images = len(os.listdir(cls.test_image_path))
        cls.n_source_vectors = len(os.listdir(cls.test_vector_path))


class TestGenerateTiles(BaseTestGeodl):
    """Unit tests for the generate_tiles method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the generate_tiles tests."""

        self.dataset.generate_tiles(dimension=self.tile_dimension,
                                    tile_path=self.test_tile_path)

        self.image_tiles_list = os.listdir(os.path.join(self.test_tile_path, "images"))
        self.label_tiles_list = os.listdir(os.path.join(self.test_tile_path, "labels"))

        self.image_tile = gdal.Open(os.path.join(self.test_tile_path, "images", self.image_tiles_list[0]))
        self.label_tile = gdal.Open(os.path.join(self.test_tile_path, "labels", self.label_tiles_list[0]))

    def test_save_location(self) -> None:
        """Test whether tiles are actually saved in the correct place."""

        self.assertGreater(len(self.image_tiles_list), 0)
        self.assertGreater(len(self.image_tiles_list), 0)

    def test_equal_tiles(self) -> None:
        """Test whether the same number of tiles are generated for the imagery and labels."""

        self.assertEqual(len(self.image_tiles_list), len(self.label_tiles_list))

    def test_number_of_bands(self) -> None:
        """Test whether tiles have correct number of bands."""

        self.assertEqual(self.image_tile.RasterCount, 3)
        self.assertEqual(self.label_tile.RasterCount, 1)

    def test_dimensions(self) -> None:
        """Test whether tiles have the correct dimensions."""

        self.assertEqual(self.image_tile.RasterXSize, 512)
        self.assertEqual(self.label_tile.RasterYSize, 512)

    def test_raster_content(self) -> None:
        """Test whether the tiles have nonzero pixels."""

        self.assertGreater(np.sum(self.image_tile.ReadAsArray()), 0)
        self.assertGreater(np.sum(self.label_tile.ReadAsArray()), 0)

    def tearDown(self) -> None:
        """Deletes the directories created by generate_tiles()."""

        shutil.rmtree(self.test_tile_path)


class TestGetLabelvectors(BaseTestGeodl):
    """Unit tests for the get_label_vectors method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the get_label_vectors tests."""

        self.dataset.get_label_vectors(osm_keys=self.test_osm_keys,
                                        save_path=self.test_vector_path)

        self.vector_folder_names = os.listdir(self.test_vector_path)
        self.n_vector_folders = len(self.vector_folder_names)
        self.n_shapefiles = 0
        
        # get number of shapefiles written
        for root, dirs, files in os.walk(self.test_vector_path):
            for file in files:
                if file.endswith(".shp"):
                    self.n_shapefiles += 1

    def test_save_location(self) -> None:
        """Test whether the vectors are saved in the correct place."""
        self.assertGreater(self.n_vector_folders, 0)

    def test_equal_number(self) -> None:
        """Test whether the correct number of vectors were downloaded."""
        self.assertEqual(self.n_source_images, self.n_shapefiles)

    def tearDown(self) -> None:
        """Deletes the directory created with get_label_vectors."""

        shutil.rmtree(self.test_vector_path)


class TestRasterizeVectors(BaseTestGeodl):
    """Unit tests for the rasterize_vectors method of the SemSeg class"""

    def setUp(self) -> None:
        """Sets up the test fixtures for the rasterize_vectors tests."""



class TestSetLabelImagery(BaseTestGeodl):
        """Test whether the set_label_imagery method works as expected."""

        def method(self):
            return None


class TestSetLabelVectors(BaseTestGeodl):
        """Test whether the set_label_vectors method works as expected."""

        def method(self):
            return None


class TestSetSourceImagery(BaseTestGeodl):
        """Test whether the set_source_imagery method works as expected."""

        def method(self):
            return None



if __name__ == "__main__":
    unittest.main()
