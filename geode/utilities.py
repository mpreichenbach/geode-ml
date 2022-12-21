# utilities.py

from numpy import arange, sum, unique
# from os import mkdir
from os.path import join, isdir, splitext
from osgeo.gdal import Dataset,  GDT_Byte, GetDriverByName, RasterizeLayer, Translate, Warp
from osgeo.ogr import DataSource
# from osgeo.osr import SpatialReference, CoordinateTransformation


# def get_osm_layer(rgb: Dataset,
#                   output_path: str,
#                   filename: str):
#     """Documentation forthcoming for this function."""
#
#     # create folder to hold polygon data
#     if not isdir(join(output_path, filename)):
#         mkdir(join(output_path, filename))
#
#     # extract bounding box coordinates for OSM query
#     ulx, xres, _, uly, _, yres = rgb.GetGeoTransform()
#     lrx = ulx + (rgb.RasterXSize * xres)
#     lry = uly + (rgb.RasterYSize * yres)
#
#     # define the source and target projections to enable conversion to lat/long coordinates
#     source = SpatialReference()
#     source.ImportFromWkt(rgb.GetProjection())
#
#     target = SpatialReference()
#     target.ImportFromEPSG(4326)
#
#     transform = CoordinateTransformation(source, target)
#
#     # get bounding box coordinates in lat/long
#     north, west, _ = list(transform.TransformPoint(ulx, uly))
#     south, east, _ = list(transform.TransformPoint(lrx, lry))

def rasterize_polygon_layer(rgb: Dataset,
                            polygons: DataSource,
                            output_path: str,
                            burn_value: int = 1,
                            no_data_value: int = 0) -> None:
    """Converts polygon vector layers into rasters of the same size as the source RGB dataset.

    Args:
        rgb: the dataset of RGB imagery;
        polygons: the dataset of the associated polygon layer;
        output_path: filepath for the output dataset;
        burn_value: the value to write for feature pixels;
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
    output_path = join(output_path)
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
                   burn_values=[burn_value])

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
