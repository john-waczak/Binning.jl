module Binning
using Flux



# define smooth function for "binning" process
function SmoothStep(x, bᵢ, bⱼ; α=5)
    # assume i < j
    return 0.5(tanh(α*(x-bᵢ))-tanh(α*(x-bⱼ))) # max val 1 min value 0 sharpness controlled by α
end


# see https://fluxml.ai/Flux.jl/stable/models/advanced/

# define the structure of the layer
struct DomainBinner{S<:AbstractArray, T<:AbstractArray, N<:Number, M<:Number} 
    x::S # input values
    b::T # vector of bin edges
    b₁::N # start bin
    bₑ::M # end bin
end

# define layer constructor
function DomainBinner(x::AbstractArray, N::Integer)
    x_min = minimum(x) 
    x_max = maximum(x) 
    
    # may wan't to specify different bin initialization according to some distribution
    Δx = (x_max-x_min)/N 
    b1 = x_min
    be = x_max
    b = Array(x_min+Δx:Δx:x_max-Δx) 
    
    return DomainBinner(x, b, b1, be)
end

# set the trainable parameters to be bin positions within the boundaries
#Flux.trainable(L::DomainBinner) = (L.b[2:end-1],)
Flux.trainable(L::DomainBinner) = (L.b,)

# define how to call layer as a function
# Note
function (BL::DomainBinner)(f::AbstractArray)
    # collect bins and sort in ascending order
    b = sort(vcat(BL.b₁,BL.b, BL.bₑ))
    x = BL.x

    # we need to treat multidimensional data carefully since this isn't a matrix operation
    # Think of rows as instances of data so that shape is (length data, num features)
    if ndims(f) == 1
        fout = [sum([f[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for i ∈ 2:length(b)]
    else
        fout = [sum([f_row[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for f_row in eachrow(f),  i ∈ 2:length(b)]
    end

    return fout
end




function getBinCenters(BL::DomainBinner)
    b = sort(vcat(BL.b₁,BL.b, BL.bₑ))
    [(b[i]+b[i-1])/2 for i ∈ 2:length(b)]
end


# not sure exactly what this is doing but it's in the docs
Flux.@functor DomainBinner



export DomainBinner
export getBinCenters
export SmoothStep

end
