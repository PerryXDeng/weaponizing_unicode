#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy
pip install opencv-python
pip install tensorflow
pip install Keras==2.4.2
pip install Pillow
pip install scikit-image
pip install efficientnet
pip install Keras-Applications
pip install cupy
deactivate
