# test_set_source_imagery.py

import numpy as np
import os
from osgeo import gdal, ogr
import shutil
from src.geodl.datasets import SemSeg
import unittest


class GeodlBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Set up test fixtures."""

        cls.test_channel_description = "RGB"
        cls.test_dataset_description = "An image over Bellingham, WA."
        cls.tile_dimension = 512
        cls.test_image_path = "test/imagery/source/"
        cls.test_polygon_path = "test/imagery/label_polygons"
        cls.test_tile_path = "test/imagery/tiles"

        cls.dataset = SemSeg(dataset_description=cls.test_dataset_description,
                             channel_description=cls.test_channel_description)


class TestGenerateTiles(GeodlBaseTest):
    """Unit tests for the generate_tiles method of the SemSeg class."""

    def setUp(self) -> None:
        # run generate_tiles() for the following tests
        self.dataset.generate_tiles(dimension=self.tile_dimension,
                                    tile_path=self.test_tile_path)

        self.image_tiles_list = os.listdir(os.path.join(self.test_tile_path, "images"))
        self.label_tiles_list = os.listdir(os.path.join(self.test_tile_path, "labels"))

        self.image_tile = gdal.Open(os.path.join(self.test_tile_path, images, image_tiles_list[0]))
        self.label_tile = gdal.Open(os.path.join(self.test_tile_path, labels, label_tiles_list[0]))

    def test_save_location(self):
        """Test whether tiles are actually saved in the correct place."""
        self.assertGreater(len(list_image_tiles), 0)
        self.assertGreater(len(list_label_tiles), 0)

    def test_equal_tiles(self):
        """Test whether the same number of tiles are generated for the imagery and labels."""
        self.assertEqual(len(list_img_tiles), len(list_label_tiles))

    def test_number_of_bands(self):
        """Test whether tiles have correct number of bands."""
        self.assertEqual(image_tile.RasterCount, 3)
        self.assertEqual(label_tile.RasterCount, 1)

    def test_dimensions(self):
        """Test whether tiles have the correct dimensions."""
        self.assertEqual(image_tile.RasterXSize, 512)
        self.assertEqual(label_tile.RasterYSize, 512)

    def test_raster_content(self):
        """Test whether the tiles have nonzero pixels."""
        self.assertGreater(np.sum(image_tile.ReadAsArray()), 0)
        self.assertGreater(np.sum(label_tile.ReadAsArray()), 0)

    def tearDown(self):
        # delete the directories created by generate_tiles()
        shutil.rmtree(self.test_tile_path)


    def test_get_label_polygons(self):
        """Test whether the get_label_polygons method works as expected."""
        return None

    def test_rasterize_labels(self):
        """Test whether the rasterize_labels method works as expected."""
        return None

    def test_resample(self):
        """Test whether the resample method works as expected."""
        return None

    def test_return_batch(self):
        """Test whether the return_batch method works as expected."""
        return None

    def test_set_label_imagery(self):
        """Test whether the set_label_imagery method works as expected."""
        return None

    def test_set_label_polygons(self):
        """Test whether the set_label_polygons method works as expected."""
        return None

    def test_set_source_imagery(self):
        """Test whether the set_source_imagery method works as expected."""
        self.dataset.set_source_imagery(path=self.test_image_path)

        self.assertEqual(self.dataset.source_path, self.test_image_path)
        self.assertListEqual(self.dataset.source_images, ["bellingham_cropped.tif"])

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove text fixtures."""

        cls.test_channel_description = None
        cls.test_dataset_description = None
        cls.test_image_path = None
        cls.tile_dimension = None
        cls.test_tile_path = None

        cls.dataset = None


if __name__ == "__main__":
    unittest.main()
