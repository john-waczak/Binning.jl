module Binning
using Flux


"""
    SmoothStep(x, bᵢ, bⱼ; α=5)

Construct a bin function with edges bᵢ and bⱼ with steepness α.


# Examples
```julia-repl
julia> SmoothStep(5, 0.0, 10.0, α=5.0)
1.0
```
"""
function SmoothStep(x, bᵢ, bⱼ; α=5)
    # assume i < j
    return 0.5(tanh(α*(x-bᵢ))-tanh(α*(x-bⱼ))) # max val 1 min value 0 sharpness controlled by α
end

# binning layer defined by following example from: 
# https://fluxml.ai/Flux.jl/stable/models/advanced/


struct DomainBinner{S<:AbstractArray, T<:AbstractArray, N<:Number, M<:Number}
    x::S # input values
    b::T # vector of bin edges
    b₁::N # start bin
    bₑ::M # end bin
end

# define layer constructor
"""
    DomainBinner(x, N)

Construct a DomainBinner layer to be used with a Flux model.
`x` are the current domain values and `N` is the desired number of bins.

# Examples
```julia-repl
julia> DomainBinner(0:1:1000, 100)
DomainBinner{StepRange{Int64, Int64}, Vector{Float64}, Int64, Int64}(0:1:1000, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0  …  900.0, 910.0, 920.0, 930.0, 940.0, 950.0, 960.0, 970.0, 980.0, 990.0], 0, 1000)
```
"""
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
# function (BL::DomainBinner)(f::AbstractArray)
#     # collect bins and sort in ascending order
#     b = sort(vcat(BL.b₁,BL.b, BL.bₑ))
#     x = BL.x

#     # we need to treat multidimensional data carefully since this isn't a matrix operation
#     # Think of rows as instances of data so that shape is (length data, num features)
#     if ndims(f) == 1
#         fout = [sum([f[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for i ∈ 2:length(b)]
#     else
#         # fout = [sum([f_row[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for   f_row in eachrow(f),  i ∈ 2:length(b)]

#         fout = [sum([f_col[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for i ∈ 2:length(b), f_col in eachcol(f)]
#         # fout = fout'

#     end

#     return fout
# end

function (BL::DomainBinner)(f::AbstractVector)
    # collect bins and sort in ascending order
    b = sort(vcat(BL.b₁,BL.b, BL.bₑ))
    x = BL.x

    fout = [sum([f[j]*SmoothStep(x[j], b[i-1], b[i]) for j ∈ 1:length(x)]) for i ∈ 2:length(b)]
    return fout
end

# function (BL::DomainBinner)(f::AbstractArray)
#     fout = BL.(eachcol(f))
# end


"""
    getBinCenters(BL:DomainBinner)

Return the current bin centers for a `DomainBinner` `BL`

# Examples
```julia-repl
julia> getBinCenters(BL)
100-element Vector{Float64}:
   5.0
  15.0
  25.0
  35.0
  45.0
  55.0
  65.0
  75.0
  85.0
  95.0
   ⋮
 915.0
 925.0
 935.0
 945.0
 955.0
 965.0
 975.0
 985.0
 995.0
```
"""
function getBinCenters(BL::DomainBinner)
    b = sort(vcat(BL.b₁,BL.b, BL.bₑ))
    [(b[i]+b[i-1])/2 for i ∈ 2:length(b)]
end


# not sure exactly what this is doing but it's in the docs
Flux.@functor DomainBinner



export SmoothStep
export DomainBinner
export getBinCenters

end
