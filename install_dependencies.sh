#!/bin/bash

# Script to install requirements
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
