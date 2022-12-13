# utils.py

import numpy as np
import os
from osgeo import gdal, ogr, osr
from pathlib import Path

def rasterize_polygon_layer(rgb: gdal.Dataset,
                            polygons: ogr.DataSource,
                            output_path: str,
                            burn_value: int = 1,
                            no_data_value: int = 0) -> None:

    # get geospatial metadata
    geo_transform = rgb.GetGeoTransform()
    projection = rgb.GetProjection()

    # get raster dimensions
    x_res = rgb.RasterXSize
    y_res = rgb.RasterYSize

    # get the polygon layer to write
    polygon_layer = polygons.GetLayer()

    # create output raster dataset
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = os.path.join(output_path)
    output_raster = gdal.GetDriverByName('GTiff').Create(output_path, x_res, y_res, 1, gdal.GDT_Byte)
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

    # write to the output file
    output_raster = None


def resample_dataset(raster: gdal.Dataset,
                     method: str,
                     target_resolution: tuple) -> SemSeg:

    raise NotImplementedError("Method \'resample\' not implemented.")

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

    return True
