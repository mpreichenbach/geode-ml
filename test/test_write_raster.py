# test_write_raster.py

from src.geodl.utils import write_raster
import numpy as np
import os
from osgeo import gdal
import unittest


class TestWriteRaster(unittest.TestCase):
    """Unit tests for the write_raster function."""

    def setUp(self):
        """Set up test fixtures."""

        self.dataset_path = "test/imagery/source/bellingham_clipped.tif"
        self.output_path = "test/imagery/source/test_rgb.tif"

        self.dataset = gdal.Open(self.dataset_path)
        write_raster(self.dataset, output_path=self.output_path)
        self.dataset0 = gdal.Open(self.output_path)

    def test_write_raster(self):
        """Test whether the write_raster functions works as expected."""

        self.assertTrue(np.all(self.dataset.ReadAsArray() == self.dataset0.ReadAsArray()))
        self.assertEqual(self.dataset.GetProjection(), self.dataset0.GetProjection())
        self.assertEqual(self.dataset.GetGeoTransform(), self.dataset0.GetGeoTransform())
        self.assertEqual(self.dataset.RasterCount, self.dataset0.RasterCount)
        self.assertEqual(self.dataset.RasterXSize, self.dataset0.RasterXSize)
        self.assertEqual(self.dataset.RasterYSize, self.dataset0.RasterYSize)

    def tearDown(self):
        # GDAL requires the loaded dataset files to be closed before the written file can be deleted
        self.dataset0 = None

        os.remove(self.output_path)


if __name__ == '__main__':
    unittest.main()
