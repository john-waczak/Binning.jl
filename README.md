# Binning Layers for Julia
## Overview
This package includes a set of `Flux.jl` compatible *binning layers*, the purpose of which is to partition a continuous variable (say frequency) into a finite number of variable size bins that can be trained via machine learning optimizers. 

Currently, there is one flavor of the binning layer: the `DomainBinner`. The `DomainBinner` partitions the input space of some function. For example, suppose we have measurements for the power spectrum of some signal as a function of frequency and time. We can use the `DomainBinner` to *learn* the ideal frequency bins to reduce the size of the dataset to only those frequency bands that are relevant to our analysis. In machine learning terms, *the `DomainBinner` performs dimensionality reduction*.  


<mark>**Note:** we should add a `Binner` layer for use on a continuous variable alone (e.g. to bin angles or something like that)</mark>


## Installation
To install the `Binning.jl` package, navigate to your desired folder. Then, activate the Julia REPL via 
```bash
$ julia
```
you should now see: 
```bash
julia>
```
press `]` to enter the `Pkg` mode. You should see
```bash
(@v1.6) pkg>
```
The `(@v1.6)` tells you the version of Julia you are currently using. If you are working on a project, it encouraged to set up a fresh environment (optional) via 
```bash
pkg> activate .
```
To add the `Binning.jl` package, simply use the `add` keyword followed by the link to *this github repository*:
```bash
pkg> add https://github.com/john-waczak/Binning.jl
```
You are now ready to utilize the machine learning layers offered by the `Binning.jl` package. 


## Examples
