How to install **geode-ml**
====================

The **geode-ml** package depends on **GDAL** and **Tensorflow** for most of its functionality. It is easiest to install 
**GDAL** using the **conda** package manager:

```
conda create -n "geode_env" python>=3.7
conda activate geode_env
conda install gdal
```

However, installing **Tensorflow** with Conda is trickier; we recommend following official documentation for installing 
the cuDNN and CUDA Toolkit libraries with the **conda** package manager (if you have a compatible GPU), and then doing

```pip install tensorflow-gpu```

After activating an environment which has both **GDAL** and **Tensorflow**, use **pip** to install **geode-ml**:

```
pip install geode-ml
```

The geode.datasets module
-------------------

The datasets module currently contains the class:

1. SemanticSegmentation
	* creates and processes pairs of imagery and label rasters for scenes

The geode.losses module
--------------------

The losses module contains custom loss functions for model training; these may be removed in the future when implemented
in Tensorflow.

The geode.metrics module
--------------------

The metrics module contains useful metrics for testing model performance.

The geode.models module
--------------------

The models module contains the classes:

1. Segmentation
	* subclass of the tensorflow.keras.Model class to be used for image segmentation
2. Unet
	* subclass of the Segmentation class which instantiates a Unet architecture.

The geode.utilities module
--------------------

The utilities module currently contains functions to process, single examples of geospatial data. The datasets module
imports these functions to apply to batches of data; however, this module exists so that methods can be used by 
themselves, without instantiating a class object from another module.
