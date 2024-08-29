include("./main.jl")


# generation_experiment("production_generation", 2, 1, 20, 2^16; discrete=false, get_data_from = "", conditional = true)

generation_experiment("cont_generation", 1, 2, 2, 20, 100; seed=120, discrete=false, get_data_from = "", conditional = true)

