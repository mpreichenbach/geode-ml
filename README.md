How to install **geode-ml**
====================

The **geode-ml** package depends on **GDAL** for most of its functionality. It is easiest to install **GDAL** using the
**conda** package manager:

```
conda create -n "geode_env" python>=3.7
conda activate geode_env
conda install gdal
```

After activating an environment which has **GDAL**, use **pip** to install **geode-ml**:

```
pip install geode-ml
```

The geode.datasets module
-------------------

The datasets module currently contains the class:

1. SemanticSegmentation
	* creates and processes pairs of imagery and label rasters for scenes

The geode.generators module
---------------------

The generators module currently contains the class:

1. TrainingGenerator
	* supplies batches of imagery/label pairs for model training
	* from_tiles() method reads from generated tile files
	* from_source() method (in development) reads from the larger source rasters

The geode.losses module
--------------------

The losses module contains custom loss functions for model training; these may be removed in the future when implemented in Tensorflow.

The geode.metrics module
--------------------

The metrics module contains useful metrics for testing model performance.

The geode.models module
--------------------

The models module contains the classes:

1. Segmentation
	* subset of the tensorflow.keras.Model class to be used for image segmentation
2. Unet
	* subset of the Segmentation class which instantiates a Unet architecture.

The geode.utilities module
--------------------

The utilities module currently contains functions to process, single examples of geospatial data. The datasets module
imports these functions to apply to batches of data; however, this module exists so they they can be used by themselves.
