#!/bin/bash

# Create virtual environment
virtualenv -p python3 venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirement.txt
