include("./main.jl")

name = "loaded_data_test"
version = 1

experiment2 = experiment(L, J, g, 0.5, (2^x for x=5:16))
experiment2(name, version; load = "production_generation")



