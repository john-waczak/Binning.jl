using Weave
using Flux
using DataFrames, CSV
using Binning
using Random:shuffle!
using BSON: @save, @load
using Plots
include("./trainModel.jl")

filename = "./evalTemplate.jl"

weave(filename; doctype="md2html", out_path=:pwd)
weave(filename; doctype="github", out_path=:pwd)
