include("./transformer.jl")

x = [0 1 1 1 ; 1 0 1 1 ; 1 1 0 1 ; 1 1 1 0]
y = [1 1 1 1 ; 0 0 0 0]

model = GeneralTransformer()
a = compute_entropy(model, x)

println(a)