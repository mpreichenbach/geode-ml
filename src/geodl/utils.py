# utils.py

from osgeo import gdal, ogr


def rasterize(vector_dataset: ogr.DataSource,
              burn_value: int = 1,
              no_data_value: int = 0) -> gdal.Dataset:

    raise NotImplementedError("Method \'rasterize\' not implemented.")

def resample(raster: gdal.Dataset,
             method: str,
             target_resolution: float) -> gdal.Dataset:

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
