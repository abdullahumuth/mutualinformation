include("./main.jl")

name = "ER_improved_unique_uniform_time_series_l20_batch128"
version = 1

experiment3 = experiment(L, J, g, 0.0:0.05:1.0, 10000)

experiment3(name, version; )