#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements_cuda101.txt
deactivate
