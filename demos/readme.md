# Binning Layer Demos 

## Directory Structure
```
├── EEG_EAR_prediction
│   ├── data
│   │   ├── O1.csv
│   │   ├── O2.csv
│   │   └── Oz.csv
│   ├── DataWrangling.ipynb
│   ├── figures
│   ├── models
│   │   ├── evalTemplate.jl
│   │   ├── evaluate.jl
│   │   ├── Oz-trainedBinner.bson
│   │   ├── Oz-trainedDense.bson
│   │   ├── Oz-trainedWhole.bson
│   │   └── trainModel.jl
│   └── Test_Model_Oz.ipynb
├── integral_test.jl
└── readme.md
```
Note, .csv files have been put into gitignore to prevent storage of data on github.

## EEG_EAR_prediction

The `EEG_EAR_prediction` folder contains scripts to train and evaluate models using the `DomainBinner` to reduce the size of EEG frequency data and use it to predict EAR  (eye aspect ratio) values . The .ipynb files are jupyter notebooks that were used for data wrangling and putting together a simple test. The `evalTemplate.jl` contains a script to load and evaluate the trained models, i.e. using them to produce a scatter plot of predicted versus true EAR values. `evaluate.jl` runs `evalTemplate.jl` using `Weave.jl` to capture the output and create .html and .md files with annotated results.



## integral_test.jl 

The file `integral_test.jl` implements a simple demonstration of the binning layer that I used to verify that gradients track through loss calculations and that the `flux.jl` optimizers are able to adjust the bin edges accordingly. 

The script generates a simple sine wave sampled between `x=0` and `x=10` with `1000` points. A `DenseBinner` is then used to reduce the length `1000` signal down to `20` binned values. The bin edges are optimized to maintain the value of the integral of the signal.  
