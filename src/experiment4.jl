include("./main.jl")


# generation_experiment("production_generation", 2, 1, 20, 2^16; discrete=false, get_data_from = "", conditional = true)

generation_experiment("continuous_production_generation", 1, 2, 2, L, 2^16; seed=120, discrete=false, get_data_from = "", conditional = true)

version = 1
load_exp = experiment(L, J, g, 0.5, (2^x for x=7:16))
load_exp("cont_loaded_data_test", version; load = "continuous_production_generation")

