include("./main.jl")

name = "magnetization_moment_of_truth_2partition_l20_batch128"
version = 1

experiment2 = experiment(L, J, g, 0.5:0.45:0.95, (2^x for x=9:16))
experiment2(name, version; )