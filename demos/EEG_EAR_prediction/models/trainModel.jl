using Flux
using Flux: @epochs
using DataFrames, CSV
using Binning
using Random:shuffle!
using BSON: @save




function getData(dataPath)
    electrodeID = split(split(dataPath, "/")[end], ".")[1]
    println(electrodeID)

    df = DataFrame(CSV.File(dataPath))
    In = df[:, Between(:Oz_freq_0Hz,:Oz_freq_250Hz)]
    Out = df[:,  :EAR_Avg]

    X = Array(In)
    X .= log10.(X) # bring the intensity into decibels
    y = Array(Out)

    return (X, y)
end

function getBatchIdxs(X, batchSize)
    idx = collect(1:batchSize-1:size(X)[1])
    push!(idx, size(X)[1])
    batches = [idx[i]:idx[i+1] for i ∈ 1:length(idx)-1]

    shuffle!(batches)

    return batches
end


function batchModelApply(X_batch, Model)
    out = Model.(eachrow(X_batch))

    # hcat to turn vector(vector) into array
    # transpose to get right shape
    return transpose(hcat(out...))
end


function loss(Xbatch, ybatch, Model)
    Flux.Losses.mse(batchModelApply(Xbatch, Model), ybatch)
end


function train!(loss, η, Model, X, y, batches, ps, outName)
    opt = Descent(η)

    for batch ∈ batches
        Xbatch = X[batch, :]
        ybatch = y[batch, :]

        gs = Flux.gradient(ps) do
            ℓ = loss(Xbatch, ybatch, Model)
            println("error: ", ℓ)
            ℓ
        end
        Flux.Optimise.update!(opt, ps, gs)
        @save outName Model
    end

end




function main()
    (X,y) = getData("../data/Oz.csv")
    batch_idx = getBatchIdxs(X, 100)

    f = range(0, stop=250, length=257)
    n_bins = 100
    B = DomainBinner(f, 100)
    D₁ = Dense(100, 100, σ)
    D₂ = Dense(100, 2)

    Model = Chain(B, D₁, D₂)

    ps₁ = params(B)
    ps₂ = params(D₁, D₂)
    ps₃ = params(Model)

    @epochs 1 train!(loss, 0.01, Model, X, y, batch_idx, ps₂, "Oz-trainedDense_3.bson")

    # reshuffle the batch
    batch_idx = getBatchIdxs(X, 100) # re
    @epochs 1 train!(loss, 0.01, Model, X, y, batch_idx, ps₁, "Oz-trainedBinner_3.bson")

    # reshuffle the batch
    batch_idx = getBatchIdxs(X, 100)
    @epochs 1 train!(loss, 0.01, Model, X, y, batch_idx, ps₃, "Oz-trainedWhole_3.bson")
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
