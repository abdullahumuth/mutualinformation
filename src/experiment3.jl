include("./main.jl")

name = "ER_improved_unique_uniform_time_series_l20_batch128"
name = "customizable_unique_uniform_time_series_l20_batch128"
version = 4

experiment3 = experiment(L, J, g, 0.0:0.05:0.95, 10000)
experiment3(name, version; unique = true, uniform = true)