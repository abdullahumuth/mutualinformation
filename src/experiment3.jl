include("./main.jl")

version = 1
load_exp = experiment(12, J, g, 0.1:0.4:0.5, (2^x for x=7:16))
load_exp("cont_l12_sample_test", version; gaussian_num = 32)
