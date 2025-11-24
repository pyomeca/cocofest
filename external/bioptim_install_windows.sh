#!/bin/bash

## Move to the external folder
cd ${0%/*}

git submodule update --recursive --init

## Move to the bioptim folder
cd bioptim

# Installing bioptim
pip install .
