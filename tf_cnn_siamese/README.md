# Tensorflow Implementation
This implementation uses the low-level Tensorflow Core api to compile
and optimize a computational graph. The low level implmentation allows
us to manually share parameters between the twin neural networks, and
provide us with greater flexibility with other lower level operations
implemented in the model, such as activation functions, batch optimization,
data transformation, and transfer learning.
## Current Progress
Model is largely implemented, except for regularizations and hyperparameter
optimization, which will wait until the actual unicode dataset is ready.
## Todo
separate exporting of twin and joined network models

xavier initialization

regularization if needed

hyperparameter optimization if needed
## Issues
TPU incompatability. Incorrectly calculated F1 score.
