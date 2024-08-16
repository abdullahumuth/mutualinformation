include("./main.jl")

name = "continuous_sample_test_t0"
version = 1

experiment3 = experiment(L, J, g, 0.0, (2^x for x=9:16))
experiment3(name, version; gaussian_num = 32)

