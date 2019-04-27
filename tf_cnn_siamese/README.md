# Tensorflow Implementation
This implementation uses the low-level Tensorflow Core api to compile
and optimize a computational graph. The low level implmentation allows
us to manually share parameters between the twin neural networks, and
provides us with greater flexibility with other lower level operations
implemented in the model, such as GPU batch optimization, transfer learning,
and custom deployment.
## Current Progress
Model is largely implemented, except for regularizations and hyperparameter
optimization, which will wait until the actual unicode dataset is ready.
## Todo
separate exporting of twin and joined network models

xavier initialization

adjusting hyperparameters for actual datasets
## Issues
TPU incompatability.
