# Spectral Analysis with Python toolkit (SAPyto 🐸)

## Purpose 


## Requirements
- numpy

## How to use it


## Regarding `SEDfit.py`

This is an old project that I wrote to fit artificially produced SEDs out
of simulations or fits from observational data.

### What does this thing do?
  - Fits data form InternalShocks
    - Separates the radiation processes (Synchrotron and IC)
    - Fits the separately
  - Integrates the fitting


### Describing each function

`Fitting(filename, magnetization, dependence, pol_order )`
 - filename:
 - magnetization:
 - dependence:
 - pol_order:

`ReadFitting(filename)`

`WritingRatios(filename, ratios, dep)`
 - filename: output file for the ratios
 - dep: values of the dependence
 - ratios: 2 x len(dep) array :: ititialy array([],[])

## To do
