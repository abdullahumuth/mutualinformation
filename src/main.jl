include("./transformer.jl")
include("./gen_samples.jl")
include("./read_wvfct.jl")

L = 12
J = -1

g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 1.0    # can be anything from collect(0:0.001:1)

# this creates the spin basis in the same convention as used by the wavefunction
# spin_basis = vec(collect(Iterators.product(fill([1,0],L)...)));

psi = read_wavefunction(L, J, g, t)
num_samples = 100

#x = rand([0,1], (4, 20)) |> todevice
#y = 2 .* rand(Float32, (2, 20)) .- 1 |> todevice


x = stack(gen_samples(psi, num_samples, L), dims = 2) |> todevice
display(x)
#convert 2*n element complex vector (psi) to n,2 matrix by separating real and imaginary parts
y = cat(transpose(real(psi)), transpose(imag(psi)), dims = 1) |> todevice

model = GeneralTransformer()
a = compute_entropy(model, x)

conditional_model = GeneralTransformer(;conditional=true)
b = compute_conditional_entropy(conditional_model, x, y)

println("Entropy: " * string(a[0]))
println("Conditional Entropy: " * string(b[0]))
println("Mutual Information: " * string(a[0] - b[0]))