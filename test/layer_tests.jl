using Binning, Test

@testset "layer tests" begin
    # Write your tests here.

    # 2x + 3y
    x = range(-10, stop=10, length=100)
    testBin = DomainBinner(x, 20)
    @test length(testBin.b) == 19 # for 20 bins there are 19 interior edges

    # test that function behaves appropriately
    @test length(testBin(rand(100))) == 20

    # test that layer can be called on data batches
    # the output need to have the right shape for subsequent
    # dense layers to be able to matrix multiply.
    @test size(testBin(rand(100, 2))) == (20, 2)


end
