# test_set_source_imagery.py

import numpy as np
import os
from osgeo import gdal
import shutil
from src.geodl.datasets import SemSeg
import unittest


class BaseTestGeodl(unittest.TestCase):

    test_channel_description = "RGB"
    test_dataset_description = "An image over Bellingham, WA."
    tile_dimension = 512
    test_image_path = "test/imagery/source/"
    test_osm_keys = ["building"]
    test_vector_path = "test/imagery/label_vectors"
    tmp = "test/imagery/tmp"
    tmp_vector_path = "test/imagery/tmp/label_vectors"
    test_raster_path = "test/imagery/label_rasters"
    tmp_raster_path = "test/imagery/tmp/label_rasters"
    tmp_tile_path = "test/imagery/tmp/tiles"
    source_imagery_names = []
    source_vector_names = []
    n_source_images = len(os.listdir(test_image_path))
    n_source_vectors = len(os.listdir(test_vector_path))

    @classmethod
    def setUpClass(cls) -> None:

        if not os.path.isdir(cls.tmp):
            os.mkdir(cls.tmp)

        cls.dataset = SemSeg(source_path=cls.test_image_path,
                             vector_path=cls.test_vector_path,
                             raster_path=cls.test_raster_path,
                             tile_dimension=cls.tile_dimension,
                             dataset_description=cls.test_dataset_description,
                             channel_description=cls.test_channel_description)


        # get the source imagery filenames
        for root, dirs, files in os.walk(cls.test_image_path):
            for file in files:
                if file.endswith(".tif"):
                    cls.source_imagery_names.append(file)

        # get the source vector filenames

        for root, dirs, files in os.walk(cls.test_vector_path):
            for file in files:
                if file.endswith(".shp"):
                    cls.source_vector_names.append(file)

        if cls.n_source_images != cls.n_source_vectors:
            raise(Exception("Different numbers of source images and source vectors."))

    @classmethod
    def tearDownClass(cls) -> None:
        """Deletes all temporary directories."""

        if os.path.isdir(cls.tmp_vector_path):
            shutil.rmtree(cls.tmp_vector_path)

        if os.path.isdir(cls.tmp_raster_path):
            shutil.rmtree(cls.tmp_raster_path)

        if os.path.isdir(cls.tmp_tile_path):
            shutil.rmtree(cls.tmp_tile_path)

        if os.path.isdir(cls.tmp):
            shutil.rmtree(cls.tmp)


class TestGenerateTiles(BaseTestGeodl):
    """Unit tests for the generate_tiles method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the generate_tiles tests."""

        if not os.path.isdir(self.tmp_tile_path):
            os.mkdir(self.tmp_tile_path)

        self.dataset.generate_tiles()

        self.image_tiles_list = os.listdir(os.path.join(self.tmp_tile_path, "imagery"))
        self.label_tiles_list = os.listdir(os.path.join(self.tmp_tile_path, "labels"))

        self.image_tile = gdal.Open(os.path.join(self.tmp_tile_path, "imagery", self.image_tiles_list[0]))
        self.label_tile = gdal.Open(os.path.join(self.tmp_tile_path, "labels", self.label_tiles_list[0]))

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


class TestGetLabelvectors(BaseTestGeodl):
    """Unit tests for the get_label_vectors method of the SemSeg class."""

    def setUp(self) -> None:
        """Sets up the test fixtures for the get_label_vectors tests."""

        if not os.path.exists(self.tmp_vector_path):
            os.mkdir(self.tmp_vector_path)

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


class TestRasterizeVectors(BaseTestGeodl):
    """Unit tests for the rasterize_vectors method of the SemSeg class"""

    def setUp(self) -> None:
        """Sets up the test fixtures for the rasterize_vectors tests."""

        # create a temporary folder
        if not os.path.exists(self.tmp_raster_path):
            os.mkdir(self.tmp_raster_path)

        # run the method on the test vector files
        self.dataset.rasterize_vectors(out_path=self.tmp_raster_path)

        # get the number of written rasters
        self.n_rasters = 0
        self.raster_names = []

        for root, dirs, files in os.walk(self.tmp_raster_path):
            for file in files:
                if file.endswith(".tif"):
                    self.n_rasters += 1
                    self.raster_names.append(file)

        # create a list of the number of pixel values in each written raster
        self.pixel_value_list = []

        for raster_name in self.raster_names:
            im = gdal.Open(os.path.join(self.tmp_raster_path, raster_name))
            n_pixel_values = np.unique(im.ReadAsArray())
            self.pixel_value_list.append(n_pixel_values)

            im.close()

        # get an object with only the unique numbers of pixel values
        self.n_pixel_values = set(self.pixel_value_list)

    def test_save_location(self) -> None:
        """Test whether the rasters are saved in the correct place."""

        self.assertGreater(self.n_rasters, 0)

    def test_equal_number(self) -> None:
        """Test whether the number of rasters equals the number of vector files."""

        self.assertEqual(self.n_source_vectors, self.n_rasters)

    def test_raster_content(self) -> None:
        """Test whether the raster has the correct burned in values."""

        self.assertEqual({2}, self.n_pixel_values)


class TestSetSourceImagery(BaseTestGeodl):
    """Unit tests for the set_source_imagery of the SemSeg class."""

    def setUp(self) -> None:
        """Set up the test fixtures for set_source_imagery tests."""

        self.dataset.set_source_imagery(self.test_image_path)

    def test_n_images(self):
        """Tests whether the number of source images is correct."""

        self.assertEqual(len(self.dataset.source_image_names), 1)

    def test_image_names(self):
        """Tests whether the image names are correct."""

        self.assertEqual(self.dataset.source_image_names[0], "bellingham_clipped.tif")

    def test_imagery_extent(self):
        """Tests whether the extent of the test image is calculated correctly."""

        test_value = self.dataset.source_metadata[self.dataset.source_image_names[0]]["data_extent"]
        true_value = 1534703.7599999998

        self.assertAlmostEqual(test_value, true_value, 2)



if __name__ == "__main__":
    unittest.main()
