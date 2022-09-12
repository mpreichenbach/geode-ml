# test_set_source_imagery.py

import numpy as np
import os
from osgeo import gdal
from src.geodl.datasets import SemSeg
import unittest


class TestSemSeg(unittest.TestCase):
    """Unit tests for the set_source_imagery method of the SemSeg class."""

    def setUp(self) -> None:
        """Set up test fixtures."""

        self.test_channel_description = "RGB"
        self.test_dataset_description = "An image over Bellingham, WA."
        self.test_image_path = "test/imagery/source/"
        self.tile_dimension = 512
        self.test_tile_path = "test/imagery/tiles"


        self.dataset = SemSeg(dataset_description=self.test_dataset_description,
                              channel_description=self.test_channel_description)

    def test_generate_tiles(self):
        self.dataset.generate_tiles(dimension=self.tile_dimension,
                                    tile_path=self.test_tile_path)

        list_image_tiles = os.listdir(os.path.join(self.test_tile_path, "images"))
        list_label_tiles = os.listdir(os.path.join(self.test_tile_path, "labels"))

        image_tile = gdal.Open(os.path.join(self.test_tile_path, images, list_image_tiles[0]))
        label_tile = gdal.Open(os.path.join(self.test_tile_path, labels, list_label_tiles[0]))

        # test whether tiles are actually saved in the correct place
        self.assertGreater(len(list_image_tiles), 0)
        self.assertGreater(len(list_label_tiles), 0)

        # test whether the same number of tiles are generated for the imagery and labels
        self.assertEqual(len(list_img_tiles), len(list_label_tiles))

        # test whether tiles have correct number of bands
        self.assertEqual(image_tile.RasterCount, 3)
        self.assertEqual(label_tile.RasterCount, 1)

        # test whether tiles have the correct dimensions
        self.assertEqual(image_tile.RasterXSize, 512)
        self.assertEqual(label_tile.RasterYSize, 512)

        # test whether the tiles have nonzero pixels
        self.assertGreater(np.sum(image_tile.ReadAsArray()), 0)
        self.assertGreater(np.sum(label_tile.ReadAsArray()), 0)

    def test_get_label_polygons(self):

    def test_rasterize_labels(self):

    def test_resample(self):

    def test_return_batch(self):

    def test_set_label_imagery(self):

    def test_set_label_polygons(self):

    def test_set_source_imagery(self):
        """Test whether the set_source_imagery method works as expected."""
        self.dataset.set_source_imagery(path=self.test_image_path)

        self.assertEqual(self.dataset.source_path, self.test_image_path)
        self.assertListEqual(self.dataset.source_images, ["bellingham_cropped.tif"])


if __name__ == "__main__":
    unittest.main()
