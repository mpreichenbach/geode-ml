Geode
=====
This package contains methods and classes to process geospatial imagery into useful training data, particularly for deep-learning applications.

Datasets
--------

The datasets module currently contains the class:

1. SemanticSegmentation
	* creates and processes pairs of imagery and label rasters for scenes

Generators
----------

The generators module currently contains the class:

1. TrainingGenerator
	* supplies batches of imagery/label pairs for model training
	* from_tiles() method reads from generated tile files
	* from_source() method (in development) reads from the larger source rasters

Utilities
---------

The utilities module currently contains functions to process geospatial imagery. The dataset classes apply have methods to apply these to batches of files.
