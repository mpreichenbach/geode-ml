# TrainingGenerator.py

from numpy.random import randint, shuffle
from numpy import array, flip, moveaxis, rot90, uint8, zeros
from os import listdir
from os.path import join
from osgeo.gdal import Open


class TrainingGenerator:
    """A class to define data generators for training deep-learning models."""

    def __init__(self, batch_size: int,
                 tile_path: None,
                 source_path: str = "",
                 labels_path: str = "",
                 perform_one_hot: bool = True,
                 n_classes: int = True,
                 rotate: bool = True,
                 flip_vertically: bool = True,
                 scale_factor: float = 1/255):

        self.batch_size = batch_size
        self.tile_path = tile_path
        self.source_path = source_path
        self.labels_path = labels_path
        self.perform_one_hot = perform_one_hot
        self.n_classes = n_classes
        self.rotate = rotate
        self.flip_vertically = flip_vertically
        self.scale_factor = scale_factor

    def from_tiles(self) -> iter:
        """Returns an iterator which yields batches of imagery/label pairs.

        Returns:
            An iterator which loads images from the pre-generated tiles.
        """

        # define the paths to find the pre-generated imagery/label tiles
        imagery_path = join(self.tile_path, "imagery")
        labels_path = join(self.tile_path, "labels")

        # get all filenames, which should be identical for imagery and labels
        imagery_filenames = listdir(imagery_path)

        # shuffle the list of filenames
        shuffle(imagery_filenames)

        # iterator which has length of filenames
        ids_train_split = range(len(imagery_filenames))

        # generate the random batches
        while True:
            for start in range(0, len(ids_train_split), self.batch_size):
                x_batch = []
                y_batch = []
                end = min(start + self.batch_size, len(ids_train_split))
                ids_train_batch = ids_train_split[start:end]

                for ID in ids_train_batch:
                    img = Open(join(imagery_path, imagery_filenames[ID])).ReadAsArray()
                    lbl = Open(join(labels_path, imagery_filenames[ID])).ReadAsArray()

                    # ensure tiles follow the channels-last convention
                    img = moveaxis(img, 0, -1)

                    # perform a random counterclockwise rotation
                    if self.rotate:
                        k_rot = randint(0, 4)
                        img = rot90(img, k=k_rot)
                        lbl = rot90(lbl, k=k_rot)

                    # perform a random vertical flip
                    if self.flip_vertically and randint(0, 2) == 1:
                        img = flip(img, axis=0)
                        lbl = flip(lbl, axis=0)

                    # rescale the imagery tile
                    img = img * self.scale_factor

                    x_batch.append(img)
                    y_batch.append(lbl)

                x_batch = array(x_batch)
                y_batch = array(y_batch)

                # perform a one-hot encoding if desired
                if self.perform_one_hot:
                    oh_y_batch = zeros(y_batch.shape + (self.n_classes,), dtype=uint8)
                    for i in range(self.n_classes):
                        oh_y_batch[:, :, :, i][y_batch == i] = 1

                    y_batch = oh_y_batch

                yield x_batch, y_batch

    def from_source(self) -> iter:
        pass
