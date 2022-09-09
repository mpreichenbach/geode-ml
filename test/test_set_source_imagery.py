import unittest
from dl_datasets import SemanticSegmentationDataset

DATASET_DESCRIPTION = "An image over Bellingham, WA."
CHANNEL_DESCRIPTION = "RGB"
TEST_IMAGE_PATH = "test/imagery/source/"


class TestSemanticSegmentation(unittest.TestCase):
    """Unit tests for the set_source_imagery method of the SemanticSegmentation class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.dataset = SemanticSegmentationDataset(dataset_description=DATASET_DESCRIPTION,
                                            channel_description=CHANNEL_DESCRIPTION)

    def test_set_source_imagery(self):
        """Test whether the set_source_imagery method works as expected."""
        self.dataset.set_source_imagery(path=TEST_IMAGE_PATH)

        self.assertEqual(self.dataset.source_path, TEST_IMAGE_PATH)
        self.assertListEqual(self.dataset.source_images, ["bellingham_cropped.tif"])


if __name__ == "__main__":
    unittest.main()
