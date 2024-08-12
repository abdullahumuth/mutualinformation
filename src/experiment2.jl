include("./main.jl")

name = "ER_improved_unique_uniform_sample_convergence_l20_batch128"
version = 1

experiment2 = experiment(L, J, g, 0.5:0.5:1.0, (2^x for x=9:16))
experiment2(name, version; )