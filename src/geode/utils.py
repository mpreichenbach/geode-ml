# utils.py

import numpy as np
import os
from osgeo import gdal, ogr, osr

def tile_raster_pair(rgb: gdal.Dataset,
                     labels: gdal.Dataset,
                     tile_dimension: int,
                     drop_single_class_tiles: bool,
                     imagery_tiles_dir: str,
                     label_tiles_dir: str,
                     filename: str):
    """Generates tiles for training data from an rgb/label pair.

    Args:
        rgb: the dataset of RGB imagery;
        labels: the dataset of single-band labeled imagery;
        tile_dimension: the pixel length of the square tiles;
        drop_single_class_tiles: whether to ignore tile pairs in which there is only a single label;
        imagery_tiles_dir: directory in which to write the RGB tiles;
        label_tiles_dir: directory in which to write the label tiles;
        filename: the name to use for the tile pairs.

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

    # get the number of tiles in each dimension
    nx_tiles = int(rgb.RasterXSize / tile_dimension)
    ny_tiles = int(rgb.RasterYSize / tile_dimension)

    # get the pixel values for the start of each tile
    x_steps = np.arange(nx_tiles) * tile_dimension
    y_steps = np.arange(ny_tiles) * tile_dimension

    # loop to generate tiles
    for i in range(len(x_steps) - 1):
        x_start = x_steps[i]
        for j in range(len(y_steps) - 1):
            y_start = y_steps[j]
            # check whether both labels exist in the label tile. Note: gives a type error without float()
            label_tile = labels.ReadAsArray(xoff=float(x_start),
                                            yoff=float(y_start),
                                            xsize=tile_dimension,
                                            ysize=tile_dimension)

            if drop_single_class_tiles and len(np.unique(label_tile)) == 1:
                continue

            # set the output paths
            tile_name = os.path.splitext(filename)[0] + "_R{row}C{col}.tif".format(row=i, col=j)
            imagery_tile_path = os.path.join(imagery_tiles_dir, tile_name)
            label_tile_path = os.path.join(label_tiles_dir, tile_name)

            # create the output imagery tile
            rgb_tile = gdal.Translate(destName=imagery_tile_path,
                                     srcDS=rgb,
                                     srcWin=[x_start, y_start, tile_dimension, tile_dimension])

            # create the output label tile
            label_tile = gdal.Translate(destName=label_tile_path,
                                      srcDS=labels,
                                      srcWin=[x_start, y_start, tile_dimension, tile_dimension])

            # close connections and write to disk
            rgb_tile = None
            label_tile = None

    # remove connections to the larger rasters
    rgb = None
    labels = None

def get_osm_layer(rgb: gdal.Dataset,
                  output_path: str,
                  filename: str):
    """Documentation forthcoming for this function."""

    # create folder to hold polygon data
    if not os.path.isdir(os.path.join(output_path, filename)):
        os.mkdir(os.path.join(output_path, filename))

    # extract bounding box coordinates for OSM query
    ulx, xres, _, uly, _, yres = rgb.GetGeoTransform()
    lrx = ulx + (rgb.RasterXSize * xres)
    lry = uly + (rgb.RasterYSize * yres)

    # define the source and target projections to enable conversion to lat/long coordinates
    source = osr.SpatialReference()
    source.ImportFromWkt(rgb.GetProjection())

    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)

    transform = osr.CoordinateTransformation(source, target)

    # get bounding box coordinates in lat/long
    north, west, _ = list(transform.TransformPoint(ulx, uly))
    south, east, _ = list(transform.TransformPoint(lrx, lry))

def rasterize_polygon_layer(rgb: gdal.Dataset,
                            polygons: ogr.DataSource,
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
    output_path = os.path.join(output_path)
    output_raster = gdal.GetDriverByName('GTiff').Create(output_path,
                                                         x_res,
                                                         y_res,
                                                         1,
                                                         gdal.GDT_Byte)
    output_raster.SetGeoTransform(geo_transform)
    output_raster.SetProjection(projection)
    band = output_raster.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    # rasterize the polygon layer
    gdal.RasterizeLayer(output_raster,
                        [1],
                        polygon_layer,
                        burn_values=[burn_value])

    # close connection and write to disk
    output_raster = None

def resample_dataset(input_path: str,
                     output_path: str,
                     target_resolution: tuple,
                     resample_algorithm: str = "cubic") -> None:
    """A wrapper of gdal.Warp, but with human-readable argument values.

    Args:
        input_path: the filepath to the input imagery;
        output_path: the filepath at which to write the resampled imagery;
        target_resolution: a tuple of the form (xRes, yRes) for target resolutions, in units of meters;
        resample_algorithm: the method used for resampling (see gdalwarp documentation for more options).

    Returns:
        None"""

    # resample imagery
    resampled = gdal.Warp(destNameOrDestDS=output_path,
                          srcDSOrSrcDSTab=input_path,
                          xRes=target_resolution[0],
                          yRes=target_resolution[1],
                          resampleAlg=resample_algorithm)

    # close connection and write to disk
    resampled = None

def write_raster(dataset: gdal.Dataset,
                 output_path: str,
                 no_data_value: int = 0) -> bool:
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
    driver = gdal.GetDriverByName("GTiff")
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
