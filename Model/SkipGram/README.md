# Model 

* This module is dedicated to generate relation embedding for every naoun pair which is above threshold count.

## Steps:

* Pickup the noun pairs & triples in preprocess splitfiles to create positive triples with varied windows [ Go to /PreprocessSplitfiles]
* Create training samples including negative samples[execute createDatasets.py]
=======
# Model Training

## Steps:

* Pickup the pairs & triples in preprocess splitfiles [ Go to /PreprocessSplitfiles]
* Create training samples [execute createDatasets.py]
* Perform training [execute training.py]
