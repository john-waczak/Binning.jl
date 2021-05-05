using Binning
using Flux
using Plots



# binning function stuff -------------------------------

x = range(-10, stop=10, length=200)
plot(x, SmoothStep.(x, 0, 4), label="binning function")

# demonstrate the action of the bin function
y = sin.(x)
b_out = [sin(xᵢ)*SmoothStep(xᵢ, 0, 4) for xᵢ ∈ x]

plot!(x, y, color=:red, label="signal")
plot!(x, b_out, color=:green, label="binned signal")

savefig("../figures/smoothstep.pdf")
savefig("../figures/smoothstep.svg")



#---- test binning layer action --------------------------
x = Array(0:.1:10)
f = sin.(x)
testBinner = DomainBinner(x, 20)
binit = testBinner.b

fnew = testBinner(f)
xnew = getBinCenters(testBinner)


p = plot()
b = sort(vcat(testBinner.b₁,testBinner.b, testBinner.bₑ))
println(size(fnew), size(b))

for i ∈ 2:length(b)
    println(i)
    plot!(p, b[i-1:i], [fnew[i-1], fnew[i-1]], color=:black, fill=(0, 0.5, :black), label="")
end
xlabel!("bin(s)")
ylabel!("count")

savefig("../figures/binPlot.pdf")


#---- test training via integration problem ----------

# set up loss function
function loss()
    bins = sort(vcat(testBinner.b₁,testBinner.b, testBinner.bₑ))
    fnew = testBinner(f)
    Δx = [bins[i]-bins[i-1] for i ∈ 2:length(bins)]
    Î = sum([fnew[i]*Δx[i] for i ∈ 1:length(fnew)])
    I = sum(0.1.*f)
    return (Î-I)^2 # square to make sure loss > 0
end

function binSpreadLoss()
    bins = sort(vcat(testBinner.b₁,testBinner.b, testBinner.bₑ))
    bmin = bins[1]
    bmax = bins[end]
    E = abs(bmax-bmin)-(testBinner.bₑ-testBinner.b₁)
end

λ=1
function composedLoss()
    return loss() + λ*binSpreadLoss()
end


println(loss())
println(binSpreadLoss())
println(composedLoss())

opt = Descent(0.0005)
data = Iterators.repeated((), 200)
ps = params(testBinner)
println(ps)

# train the model!
Flux.train!(composedLoss, ps, data, opt, cb=()->println(composedLoss()))

println(composedLoss())


#p1 = plot()
b = sort(vcat(testBinner.b₁,testBinner.b, testBinner.bₑ))

for i ∈ 2:length(b)
    plot!(p, b[i-1:i], [fnew[i-1], fnew[i-1]], color=:blue, fill=(0, 0.5, :blue), label="")
end
xlabel!("bin(s)")
ylabel!("count")
savefig("../figures/trained_bin_comparison.pdf")
display(p)
