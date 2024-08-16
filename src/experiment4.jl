include("./main.jl")

name = "generate_samples_first"

generation_experiment(name, 2, 20, 100; generate = true, conditional = true)