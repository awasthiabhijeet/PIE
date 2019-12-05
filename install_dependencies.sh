#!/bin/bash

# Script to install requirements
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
