include("./main.jl")

version = 1

experiment2 = experiment(12:8:20, J, g, 0.1:0.8:0.9, (2^x for x=5:16))
experiment2("noise0.001_discrete_sample", version; noise = 0.001)
experiment2("noise0.01_discrete_sample", version; noise = 0.01)
experiment2("noise0.1_discrete_sample", version; noise = 0.1)


