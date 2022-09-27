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
        cls.test_channel_description = "RGB"
        cls.test_dataset_description = "An image over Bellingham, WA."
        cls.tile_dimension = 512
        cls.test_image_path = "test/imagery/source/"
        cls.test_osm_keys = ["building"]
        cls.test_vector_path = "test/imagery/label_vectors"
        cls.test_raster_path = "test/imagery/label_rasters"
        cls.test_tile_path = "test/imagery/tiles"

        cls.dataset = SemSeg(dataset_description=cls.test_dataset_description,
                             channel_description=cls.test_channel_description)

        cls.n_source_images = len(os.listdir(cls.test_image_path))
        cls.n_source_vectors = len(os.listdir(cls.test_vector_path))

        # get the source imagery filenames
        cls.source_imagery_names = []

        for root, dirs, files in os.walk(cls.test_vector_path):
            for file in files:
                if file.endswith(".shp"):
                    cls.source_imagery_names.append(file)

        # get the source vector filenames
        cls.source_vector_names = []

        for root, dirs, files in os.walk(cls.test_vector_path):
            for file in files:
                if file.endswith(".shp"):
                    cls.source_vector_names.append(file)

        if len(cls.source_imagery_names) != len(cls.source_vector_names):
            raise(Exception("Different numbers of source images and source vectors."))

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

        # create a temporary directory
        self.test_out_path = os.path.join(self.test_vector_path, "tmp")

        if not os.path.exists(self.test_out_path):
            os.mkdir(self.test_out_path)

        # run the method on the test vector files
        self.dataset.rasterize_vectors(out_path=self.test_out_path)

        # get the number of written rasters
        self.n_rasters = 0
        self.raster_names = []

        for root, dirs, files in os.walk(self.test_out_path):
            for file in files:
                if file.endswith(".tif"):
                    self.n_rasters += 1
                    self.raster_names.append(file)

        # create a list of the number of pixel values in each written raster
        self.pixel_value_list = []

        for raster_name in self.raster_names:
            im = gdal.Open(os.path.join(self.test_out_path, raster_name))
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

    def tearDown(self) -> None:
        """Deletes the temporary directory created to hold the rasters."""

        shutil.rmtree(self.test_out_path)


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

        test_value = [*self.dataset.source_extents.values()][1]
        true_value = 1534703.7599999998

        self.assertAlmostEqual(test_value, true_value, 2)



if __name__ == "__main__":
    unittest.main()
