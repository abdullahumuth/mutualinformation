include("./main.jl")

name = "ER_improved_moment_of_truth_batch128_1partition"
version = 1
experiment4 = experiment(L, J, g, 0.0:1.0:1.0, (2^x for x=9:16))
experiment4(name, version; )