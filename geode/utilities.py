# utilities.py

from numpy import arange, argmax, expand_dims, moveaxis, pad, squeeze, sum, uint8, unique, zeros
from os.path import join, splitext
from osgeo.gdal import Dataset,  GDT_Byte, GetDriverByName, RasterizeLayer, Translate, Warp
from osgeo.ogr import DataSource
from tensorflow.keras import Model


def convert_labels_to_one_hots(label_array, n_classes):
    """Converts integer labels with shape (n_images, height, width) to one-hot encodings."""

    enc = zeros(label_array.shape + (n_classes,), dtype=uint8)

    for i in range(n_classes):
        enc[:, :, i][label_array == i] = 1

    return enc

def convert_vectors_to_labels(oh_array):
    output = argmax(oh_array, axis=-1).astype(uint8)

    return output

def predict_raster(input_dataset: Dataset,
                   model: Model,
                   output_path: str,
                   tile_dim: int=1024) -> None:
        """Performs the inference using the supplied model, and reads tiles one-by-one from the input dataset, to avoid
        overrunning the RAM by reading the entire raster into memory. Performs inference on four regions, denoted by A,
        B, C, and D.

        Args:
            input_dataset: the raster to perform inference on;
            model: the model which performs inference;
            output_path: the path at which to save the predicted raster;
            tile_dim: the window size for inference.

        Returns:
            None
        """

        # get the input raster dimensions and tile size
        input_width = input_dataset.RasterXSize
        input_height = input_dataset.RasterYSize
        n_bands = input_dataset.RasterCount

        # get the dimensions of Region A
        n_col = int(input_width / tile_dim)
        n_row = int(input_height / tile_dim)
        a_width = tile_dim * n_col
        a_height = tile_dim * n_row

        # get the dimensions of Region B
        b_width = input_width - a_width
        b_tile_pads = ((0, 0), (0, 0), (0, tile_dim - b_width))

        # get the dimensions of Region C
        c_height = input_height - a_height
        c_tile_pads = ((0, 0), (0, tile_dim - c_height), (0, 0))

        # get the dimensions of Region D
        d_width = b_width
        d_height = c_height
        d_tile_pads = ((0, 0), (0, tile_dim - c_height), (0, tile_dim - b_width))

        # initialize the predictions array
        pred = zeros((input_height, input_width), dtype=uint8)

        # perform inference over Region A
        for row in range(n_row):
            y_start = row * tile_dim
            for col in range(n_col):
                x_start = col * tile_dim
                tile = input_dataset.ReadAsArray(xoff=int(x_start),
                                                 yoff=int(y_start),
                                                 xsize=tile_dim,
                                                 ysize=tile_dim)

                # reshape to (1, height, width, channels) format for model.predict method
                tile = expand_dims(moveaxis(tile,
                                            source=0,
                                            destination=2),
                                   axis=0)
                pred_tile = argmax(squeeze(model.predict(tile)), axis=-1).astype(uint8)

                # store the predictions into the initialized array
                pred[y_start:(y_start + tile_dim), x_start:(x_start + tile_dim)] = pred_tile

        # perform inference over Region B
        if b_width == 0:
            pass
        else:
            x_start = input_width - tile_dim
            for row in range(n_row):
                y_start = row * tile_dim
                tile = input_dataset.ReadAsArray(xoff=int(x_start),
                                                 yoff=int(y_start),
                                                 xsize=tile_dim,
                                                 ysize=tile_dim)

                # pad values over right edge, reshape tile to correct format
                tile = pad(tile,
                           pad_width=b_tile_pads,
                           mode='reflect')
                tile = tile[:, :, -tile_dim:]
                tile = expand_dims(moveaxis(tile,
                                            source=0,
                                            destination=2),
                                   axis=0)
                tile = tile.reshape((1, tile_dim, tile_dim, n_bands))

                # perform inference, remove padding, and update the full prediction array
                pred_tile = argmax(squeeze(model.predict(tile)), axis=-1).astype(uint8)
                pred_tile = pred_tile[:, 0:b_width]
                pred[y_start:(y_start + tile_dim), (input_width - b_width):input_width] = pred_tile

        # perform inference over Region C
        if c_height == 0:
            pass
        else:
            y_start = input_height - tile_dim
            for col in range(n_col):
                x_start = col * tile_dim
                tile = input_dataset.ReadAsArray(xoff=int(x_start),
                                                 yoff=int(y_start),
                                                 xsize=tile_dim,
                                                 ysize=tile_dim)

                # pad values over the bottom edge, reshape tile to correct format
                tile = pad(tile,
                           pad_width=c_tile_pads,
                           mode='reflect')
                tile = tile[:, -tile_dim:, :]
                tile = expand_dims(moveaxis(tile,
                                            source=0,
                                            destination=2),
                                   axis=0)
                tile = tile.reshape((1, tile_dim, tile_dim, n_bands))

                # perform inference
                pred_tile = argmax(squeeze(model.predict(tile)), axis=-1).astype(uint8)
                pred_tile = pred_tile[0:c_height, :]
                pred[(input_height - c_height):input_height, x_start:(x_start + tile_dim)] = pred_tile

        # perform inference over Region D
        if b_width == 0 or c_height == 0:
            pass
        else:
            x_start = input_width - tile_dim
            y_start = input_height - tile_dim
            tile = input_dataset.ReadAsArray(xoff=int(x_start),
                                             yoff=int(y_start),
                                             xsize=tile_dim,
                                             ysize=tile_dim)

            # pad values over the right and bottom edges, reshape to correct tile format
            tile = pad(tile,
                       pad_width=d_tile_pads,
                       mode='reflect')
            tile = tile[:, -tile_dim:, -tile_dim:]
            tile = expand_dims(moveaxis(tile,
                                        source=0,
                                        destination=2),
                               axis=0)
            tile = tile.reshape((1, tile.shape[1], tile.shape[2], n_bands))

            # perform inference
            pred_tile = argmax(squeeze(model.predict(tile)), axis=-1).astype(uint8)
            pred_tile = pred_tile[0:c_height, 0:b_width]
            pred[(input_height - d_height):input_height, (input_width - d_width):input_width] = pred_tile

        # check that the output_path specifies a tif file:
        if splitext(output_path)[1] == ".tif":
            pass
        else:
            raise Exception("Please specify a tif file in the output_path argument.")

        # set up the metadata and write the predicted dataset
        driver = GetDriverByName("GTiff")
        driver.Register()
        output_dataset = driver.Create(output_path,
                                       xsize=input_width,
                                       ysize=input_height,
                                       bands=1,
                                       eType=input_dataset.GetRasterBand(1).DataType)

        output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
        output_dataset.SetProjection(input_dataset.GetProjection())
        output_band = output_dataset.GetRasterBand(1)
        output_band.WriteArray(pred)
        output_band.SetNoDataValue(0)
        output_band.FlushCache()

        # without the following lines, the arrays won't actually be written into the tif file
        output_band = None
        output_dataset = None

def rasterize_polygon_layer(rgb: Dataset,
                            polygons: DataSource,
                            output_path: str,
                            burn_attribute: str,
                            no_data_value: int = 0) -> None:
    """Converts polygon vector layers into rasters of the same size as the source RGB dataset.

    Args:
        rgb: the dataset of RGB imagery;
        polygons: the dataset of the associated polygon layer;
        output_path: filepath for the output dataset;
        burn_attribute: the column name in the attribute table of values to write to the raster;
        no_data_value: the value to write for non-feature pixels.

    Returns:
        None"""

    # get geospatial metadata
    geo_transform = rgb.GetGeoTransform()
    projection = rgb.GetProjection()

    # get raster dimensions
    x_res = rgb.RasterXSize
    y_res = rgb.RasterYSize

    # get the polygon layer to write
    polygon_layer = polygons.GetLayer()

    # create output raster dataset
    output_raster = GetDriverByName('GTiff').Create(output_path,
                                                    x_res,
                                                    y_res,
                                                    1,
                                                    GDT_Byte)
    output_raster.SetGeoTransform(geo_transform)
    output_raster.SetProjection(projection)
    band = output_raster.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    # rasterize the polygon layer
    RasterizeLayer(output_raster,
                   [1],
                   polygon_layer,
                   options=["ATTRIBUTE={a}".format(a=burn_attribute)])

    # close connection and write to disk
    output_raster = None


def resample_dataset(input_path: str,
                     output_path: str,
                     target_resolutions: tuple,
                     resample_algorithm: str = "cubic") -> None:
    """A wrapper of gdal.Warp, but with human-readable argument values.

    Args:
        input_path: the filepath to the input imagery;
        output_path: the filepath at which to write the resampled imagery;
        target_resolutions: a tuple of the form (xRes, yRes) for target resolutions, in units of meters;
        resample_algorithm: the method used for resampling (see gdalwarp documentation for more options).

    Returns:
        None"""

    # resample imagery
    resampled = Warp(destNameOrDestDS=output_path,
                     srcDSOrSrcDSTab=input_path,
                     xRes=target_resolutions[0],
                     yRes=target_resolutions[1],
                     resampleAlg=resample_algorithm)

    # close connection and write to disk
    resampled = None


def tile_raster_pair(rgb: Dataset,
                     labels: Dataset,
                     tile_dimension: int,
                     imagery_tiles_dir: str,
                     label_tiles_dir: str,
                     filename: str,
                     label_proportion: float=0.2):
    """Generates tiles for training data from an rgb/label pair.

    Args:
        rgb: the dataset of RGB imagery;
        labels: the dataset of single-band labeled imagery;
        tile_dimension: the pixel length of the square tiles;
        imagery_tiles_dir: directory in which to write the RGB tiles;
        label_tiles_dir: directory in which to write the label tiles;
        filename: the name to use for the tile pairs;
        label_proportion: the minimum proportion which any single class must have per tile.

    Returns:
        None

    Raises:
        Exception: if dimensions of rgb and labels do not match."""

    # boolean values for whether dimensions are equal
    bool_x = rgb.RasterXSize == labels.RasterXSize
    bool_y = rgb.RasterYSize == labels.RasterYSize

    # test to ensure input imagery have the same dimensions
    if bool_x or bool_y:
        pass
    else:
        raise Exception("Input imagery does not have the same dimensions.")

    # get the number of pixels per tile
    n_pixels = tile_dimension ** 2

    # get the number of tiles in each dimension
    nx_tiles = int(rgb.RasterXSize / tile_dimension)
    ny_tiles = int(rgb.RasterYSize / tile_dimension)

    # get the pixel values for the start of each tile
    x_steps = arange(nx_tiles) * tile_dimension
    y_steps = arange(ny_tiles) * tile_dimension

    # set a counter to name tiles
    counter = 0

    # loop to generate tiles
    for i in range(len(x_steps) - 1):
        x_start = x_steps[i]
        for j in range(len(y_steps) - 1):
            y_start = y_steps[j]

            # read the RGB tile
            rgb_tile = rgb.ReadAsArray(xoff=float(x_start),
                                       yoff=float(y_start),
                                       xsize=tile_dimension,
                                       ysize=tile_dimension)

            # sum across the channel (GDAL arrays are channel-first)
            band_sum = sum(rgb_tile, axis=0)

            # skip tiles which have a NoData pixel (which are read as 0 in each channel)
            if 0 in unique(band_sum):
                continue

            # read the corresponding labels tile
            label_tile = labels.ReadAsArray(xoff=float(x_start),
                                            yoff=float(y_start),
                                            xsize=tile_dimension,
                                            ysize=tile_dimension)

            # get positive label proportion
            tile_proportion = sum(label_tile) / n_pixels

            if tile_proportion < label_proportion or tile_proportion > (1 - label_proportion):
                continue

            # set the output paths
            tile_name = splitext(filename)[0] + "_{counter}.tif".format(counter=counter)
            imagery_tile_path = join(imagery_tiles_dir, tile_name)
            label_tile_path = join(label_tiles_dir, tile_name)

            # create the output imagery tile
            rgb_tile = Translate(destName=imagery_tile_path,
                                 srcDS=rgb,
                                 srcWin=[x_start, y_start, tile_dimension, tile_dimension])

            # create the output label tile
            label_tile = Translate(destName=label_tile_path,
                                   srcDS=labels,
                                   srcWin=[x_start, y_start, tile_dimension, tile_dimension])

            # increment the counter by 1
            counter += 1

            # close connections and write to disk
            rgb_tile = None
            label_tile = None

    # remove connections to the larger rasters
    rgb = None
    labels = None


def write_raster(dataset: Dataset,
                 output_path: str,
                 no_data_value: int = 0) -> None:
    """Writes the predicted array, with correct metadata values, to a tif file.

    Args:
        dataset: the gdal.Dataset object to write to a tif file,
        output_path: the file in which to write the predictions,
        no_data_value: the value to assign to no_data entries of the raster

    Returns:
        None
    """

    # check that the output_path specifies a tif file:
    if output_path[-3:] == "tif":
        pass
    else:
        raise Exception("Please specify a tif file in the output_path argument.")

    # set up the metadata and write the predicted dataset
    driver = GetDriverByName("GTiff")
    driver.Register()
    output_dataset = driver.Create(output_path,
                                   xsize=dataset.RasterXSize,
                                   ysize=dataset.RasterYSize,
                                   bands=dataset.RasterCount,
                                   eType=dataset.GetRasterBand(1).DataType)

    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    output_dataset.SetProjection(dataset.GetProjection())
    for band in range(dataset.RasterCount):
        output_band = output_dataset.GetRasterBand(band + 1)
        output_band.WriteArray(dataset.GetRasterBand(band + 1).ReadAsArray())
        output_band.SetNoDataValue(no_data_value),
        output_band.FlushCache()
        output_band = None

    output_dataset = None
