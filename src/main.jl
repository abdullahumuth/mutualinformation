include("./transformer.jl")
include("./gen_samples.jl")
include("./read_wvfct.jl")
using Plots


L = 12
J = -1

g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 1.0   # can be anything from collect(0:0.001:1)

# this creates the spin basis in the same convention as used by the wavefunction
# spin_basis = vec(collect(Iterators.product(fill([1,0],L)...)));

psi = read_wavefunction(L, J, g, t)
num_samples = 1000

#x = rand([0,1], (4, 20)) |> todevice
#y = 2 .* rand(Float32, (2, 20)) .- 1 |> todevice


x = stack(gen_samples(psi, num_samples, L), dims = 2) |> todevice
display(x)
#convert 2*n element complex vector (psi) to n,2 matrix by separating real and imaginary parts
psi_vectorized = cat(transpose(real(psi)), transpose(imag(psi)), dims = 1) |> todevice

y = mapslices(x, dims = 1) do xi
    index = parse(Int, join(string.(Int.(xi .== 1))), base=2)
    return psi_vectorized[:,index+1]
    end

display(y)

println("Shape of x: ", size(x))
println("Shape of y: ", size(y))

model = GeneralTransformer()
a = compute_entropy(model, x)

println("Entropy: " * string(a[1]))

p = plot(a[2].train_losses, label="train")
plot!(a[2].test_losses, label="test")


conditional_model = GeneralTransformer(conditional=true)
b = compute_conditional_entropy(conditional_model, x, y) 

println("Conditional Entropy: " * string(b[1]))

# For the causal self-attention layer of conditional entropy estimator, 
# I may use the same parameters from the entropy estimator and see how it performs.

println("Mutual Information: " * string(a[1] - b[1]))

# println("Train Losses: ", b[2].train_losses)
# println("Test Losses: ", b[2].test_losses)

p2 = plot(b[2].train_losses, label="train")
plot!(b[2].test_losses, label="test")

display(plot(p))
display(plot(p2))

