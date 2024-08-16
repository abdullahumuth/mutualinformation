include("./main.jl")

name = "better_conditionals"
version = 1

experiment2 = experiment(L, J, g, 0.5, (2^x for x=9:16))
experiment2(name, version;)



