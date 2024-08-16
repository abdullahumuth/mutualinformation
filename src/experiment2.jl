include("./main.jl")

name = "continuous_sanity_test"
version = 1

experiment2 = experiment(L, J, g, 0.5, (10000 for x=9:16))
experiment2(name, version; gaussian_num = 32)



