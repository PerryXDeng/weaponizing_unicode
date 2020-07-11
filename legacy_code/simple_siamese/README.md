# Numpy Implementation
this directory contains an implementation of the siamese twin neural network architecture in numpy
## Current Progress
implementation abandoned. moved on to more promising alternatives in tensorflow or pytorch to for ease of development
and deployment of convolution layers
## Issues
1. numpy numeric instability

gradients from backpropagation are sometiems the exact negative of what they really are

2. sigmoid activation

leads to vanishing gradients after a few layers. developer (me) not familiar with numpy implementatation of relu
activation and gradients

3. lack of convolution

convolutions are not implemented -> does not work well with images

4. no normalization
