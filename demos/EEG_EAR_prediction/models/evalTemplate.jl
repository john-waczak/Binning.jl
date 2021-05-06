#'---
#'title : Binning Layer EAR Model Evaluation
#'author : John Waczak
#'date : 5-4-2021
#'---



#'Summon the needed Packages as well as functions from `trainModel.jl`
using Flux
using DataFrames, CSV
using Binning
using Random:shuffle!
using BSON: @save, @load
using Plots
include("./trainModel.jl")



#'Grab the data
dataPath = "../data/Oz.csv"
(Xtrain, Ytrain) = getData(dataPath)

#'Load the model
@load "./Oz-trainedWhole.bson" Model
#' Apply the model to the entire dataset
Y_model = batchModelApply(Xtrain, Model)
Y_truth = Ytrain

#'Create a scatter plot of the result
p = plot(Y_model, Y_truth, seriestype = :scatter, color=:indigo, xlabel="model", ylabel="truth", title="Avg EAR", label="")
plot!(p, [-10, 10], [-10, 10], color=:red, linestyle=:dash, label="1:1")
xlims!(p, min(Y_model...), max(Y_model...))
ylims!(p, min(Y_truth...), max(Y_truth...))


#'Save the resulting figure
savefig(p, "../figures/Avg-EAR.png")


#print out the bins
println(Model[1].b)
