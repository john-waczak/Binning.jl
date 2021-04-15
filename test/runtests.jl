using Binning, SafeTestsets, Test


# in the root of the directory run:
# pkg> test BinningLayer
# to run all of the tests.

@time begin

@time @safetestset "layer tests" begin include("layer_tests.jl") end

end
