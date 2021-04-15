# Binning Layers for Julia
## Overview
This package includes a set of `Flux.jl` compatible *binning layers*, the purpose of which is to partition a continuous variable (say frequency) into a finite number of variable size bins that can be trained via machine learning optimizers. 


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
