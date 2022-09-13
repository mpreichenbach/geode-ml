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
        cls.test_osm_vector_path = "test/imagery/osm_label_polygons"
        cls.test_vector_path = "test/imagery/label_polygons"
        cls.test_raster_path = "test/imagery/label_rasters"
        cls.test_tile_path = "test/imagery/tiles"

        cls.dataset = SemSeg(dataset_description=cls.test_dataset_description,
                             channel_description=cls.test_channel_description)

        cls.n_source_images = len(os.listdir(cls.test_image_path))
        cls.n_source_vectors = len(os.listdir(cls.source_polygon_path))


class TestGenerateTiles(BaseTestGeodl):
    """Unit tests for the generate_tiles method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the generate_tiles tests."""

        self.dataset.generate_tiles(dimension=self.tile_dimension,
                                    tile_path=self.test_tile_path)

        self.image_tiles_list = os.listdir(os.path.join(self.test_tile_path, "images"))
        self.label_tiles_list = os.listdir(os.path.join(self.test_tile_path, "labels"))

        self.image_tile = gdal.Open(os.path.join(self.test_tile_path, images, image_tiles_list[0]))
        self.label_tile = gdal.Open(os.path.join(self.test_tile_path, labels, label_tiles_list[0]))

    def test_save_location(self) -> None:
        """Test whether tiles are actually saved in the correct place."""

        self.assertGreater(len(list_image_tiles), 0)
        self.assertGreater(len(list_label_tiles), 0)

    def test_equal_tiles(self) -> None:
        """Test whether the same number of tiles are generated for the imagery and labels."""

        self.assertEqual(len(list_img_tiles), len(list_label_tiles))

    def test_number_of_bands(self) -> None:
        """Test whether tiles have correct number of bands."""

        self.assertEqual(image_tile.RasterCount, 3)
        self.assertEqual(label_tile.RasterCount, 1)

    def test_dimensions(self) -> None:
        """Test whether tiles have the correct dimensions."""

        self.assertEqual(image_tile.RasterXSize, 512)
        self.assertEqual(label_tile.RasterYSize, 512)

    def test_raster_content(self) -> None:
        """Test whether the tiles have nonzero pixels."""

        self.assertGreater(np.sum(image_tile.ReadAsArray()), 0)
        self.assertGreater(np.sum(label_tile.ReadAsArray()), 0)

    def tearDown(self) -> None:
        """Deletes the directories created by generate_tiles()."""

        shutil.rmtree(self.test_tile_path)


class TestGetLabelPolygons(BaseTestGeodl):
    """Unit tests for the get_label_polygons method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the get_label_polygons tests."""

        self.dataset.get_label_polygons(osm_keys=self.test_osm_keys,
                                        save_path=self.test_vector_path)

        self.vector_folder_names = os.listdir(self.test_vector_path)
        self.n_vector_folders = len(self.vector_folder_names)
        self.n_shapefiles = 0
        
        # get number of shapefiles written
        for root, dirs, files in os.walk(self.test_vector_path):
            for file in files:
                if file.endswith(".shp"):
                    self.n_polygon_files += 1

    def test_save_location(self) -> None:
        """Test whether the polygons are saved in the correct place."""
        self.assertGreater(self.n_vector_folders, 0)

    def test_equal_number(self) -> None:
        """Test whether the correct number of polygons were downloaded."""
        self.assertEqual(self.n_source_images, self.n_shapefiles)

    def tearDown(self) -> None:
        """Deletes the directory created with get_label_polygons."""

        shutil.rmtree(self.test_vector_path)


class TestRasterizeLabels(BaseTestGeodl):
    """Unit tests for the rasterize_labels method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the rasterize_labels tests."""

        self.dataset.rasterize_labels(save_path=self.test_raster_path)
        self.n_rasters = len(os.listdir(self.test_raster_path))

    def test_save_location(self) -> None:
        """Test whether the rasters were saved in the correct place."""
        self.assertGreater(self.n_rasters, 0)

    def test_equal_number(self):
        """Test whether the same number of rasters were created as there are polygons."""
        self.assertEqual(self.n_source_images, self.n_rasters)

    def tearDown(self) -> None:
        """Deletes the directory created by rasterize_labels."""


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