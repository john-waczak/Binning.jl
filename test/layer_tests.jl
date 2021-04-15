using Binning, Test

@testset "layer tests" begin
    # Write your tests here.

    # 2x + 3y
    x = -10:0.05:10
    testBin = BinningLayer(x, 20)
    @test length(testBin.b) == 19 # for 20 bins there are 19 interior edges
end
