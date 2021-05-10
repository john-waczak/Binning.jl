# Binning Layers for Julia


## Overview
`Binning.jl` is an extension for `Flux.jl` to include a new neural-network compatible layer designed for dimensionality reduction. The new layer, called a `DomainBinner`, reduces the size of input data by resampling the data by partition the domain into N variable width bins. The collection of N-1 interior bin edges form the set of trainable parameters for the layer, allowing the user to optimize the bin edges to find the ideal *binning* of the data for some desired loss function. 


## Installation
To install the `Binning.jl` package, navigate to your desired folder. Then, activate the Julia REPL via 
```bash
$ julia
```
you should now see: 
```julia-repl
julia>
```
press `]` to enter the package manager mode. You should see
```julia-repl
(@v1.6) pkg>
```
where the `(@v1.6)` specifies the current version of Julia set by your environment. If you are working on a project, it encouraged to set up a fresh environment (optional) via 
```julia-repl
pkg> activate .
```
To add the `Binning.jl` package, simply use the `add` keyword followed by the link to *this github repository*:
```julia-repl
pkg> add https://github.com/john-waczak/Binning.jl
```
You are now ready to utilize the machine learning layers offered by the `Binning.jl` package. To make the binning layer accessible to your code, simply add
```julia
using Binning
```
to the top of your script. 

## File Structure
```
├── demos
│   ├── EEG_EAR_prediction
│   │   ├── DataWrangling.ipynb
│   │   ├── figures
│   │   ├── models
│   │   │   ├── evalTemplate.jl
│   │   │   ├── evaluate.jl
│   │   │   ├── Oz-trainedBinner.bson
│   │   │   ├── Oz-trainedDense.bson
│   │   │   ├── Oz-trainedWhole.bson
│   │   │   └── trainModel.jl
│   │   └── Test_Model_Oz.ipynb
│   ├── integral_test.jl
│   └── readme.md
├── LICENSE
├── notebooks
│   ├── Binning Layer Development.ipynb
│   ├── images
│   │   ├── Avg-EAR3.png
│   │   ├── binningAction.svg
│   │   ├── binningModel.svg
│   │   ├── environmentActivation.gif
│   │   ├── fluxGithub.png
│   │   ├── helpDemo.gif
│   │   ├── packageInstall2.gif
│   │   ├── packageInstall.gif
│   │   ├── tensorflowGithub.png
│   │   ├── testSuite.gif
│   │   ├── trained_bin_comparison.svg
│   │   ├── travis.png
│   │   └── weaveDemo.png
│   ├── PRC-Presentation.ipynb
│   ├── PRC-Presentation.md
│   └── PRC-Presentation.slides.html
├── Project.toml
├── README.md
├── src
│   └── Binning.jl
└── test
    ├── layer_tests.jl
    └── runtests.jl
```

`/src/` contains the source code for the binning layer module. The main feature of the module is the `DomainBinner` which adds a new `Flux.jl` compatible layer to be used within neural networks.  `/tests/` contains test functions (need to expand) that should be run whenever changes are made to the source code. To run the tests, open a julia repl in the root directory and activate the Binning layer environment. (press `]` to enter the package manager)
```julia-repl 
(@v1.6) pkg> activate .
  Activating environment at `~/gitRepos/Binning.jl/Project.toml`

(Binning) pkg> 

```
then, execute the tests via 
```julia-repl
(Binning) pkg> test Binning
```

The `/demos/` folder contains two demos: the `integral_test.jl` and `EEG_EAR_prediction`. See the [readme.md](./demos/readme.md) file for more details. 

The `/notebooks/` directory contains jupyter notebooks used to develop the binning layer. The project presentation for practical research computing was generated using `/notebooks/PRC-Presentation.ipynb`. To generate the slideshow using websides, navigate to the directory and enter
```bash
$ jupyter nbconvert ./PRC-Presentation.ipynb --to slides --post serve SlidesExporter.reveal_theme=serif SlidesExporter.reveral_scroll=True SlidesExporter.reveal_transition=none
```
A markdown version of the slides can be found [here](./notebooks/PRC-Presentation.md)
