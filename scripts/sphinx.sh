#!/bin/sh

# BLT requires a hard-coded path for sphix, but sometimes
# virtual environments are helpful. This script provides 
# an intermediate so that the sphinx used to generate
# documentation is chosen at build time.
# Also, Quartz's default sphinx-build doesn't have the right 
# packages to build the docs, but for some reason
# python -m sphinx does.


python -m sphinx $@