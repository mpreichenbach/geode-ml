# test_set_source_imagery.py

import unittest
from dl_datasets import SemanticSegmentationDataset


class TestSemanticSegmentation(unittest.TestCase):
    """Unit tests for the set_source_imagery method of the SemanticSegmentation class."""

    def setUp(self) -> None:
        """Set up test fixtures."""

        self.test_dataset_description = "An image over Bellingham, WA."
        self.test_channel_description = "RGB"
        self.test_image_path = "test/imagery/source/"

        self.dataset = SemanticSegmentationDataset(dataset_description=self.test_dataset_description,
                                                   channel_description=self.test_channel_description)

    def test_set_source_imagery(self):
        """Test whether the set_source_imagery method works as expected."""
        self.dataset.set_source_imagery(path=self.test_image_path)

        self.assertEqual(self.dataset.source_path, self.test_image_path)
        self.assertListEqual(self.dataset.source_images, ["bellingham_cropped.tif"])


if __name__ == "__main__":
    unittest.main()
