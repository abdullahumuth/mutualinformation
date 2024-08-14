include("./main.jl")

name = "magnetization_moment_of_truth_2partition_l20_batch128"
version = 3

experiment2 = experiment(L, J, g, 0.5, (2^x for x=10:16))
experiment2(name, version; uniform=true, fake=true, shuffle=false)