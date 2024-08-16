include("./main.jl")


generation_experiment("production_generation", 2, 20, 2^16; get_data_from = "", conditional = true)
