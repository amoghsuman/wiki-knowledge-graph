#!/bin/bash

echo "Downloading spaCy model..."
python -m spacy download en_core_web_lg

echo "Installing coreferee separately..."
pip install coreferee==1.1.3
python -m coreferee install en
