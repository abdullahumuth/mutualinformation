include("./main.jl")
include("plot_from_json.jl")

# generation_experiment("production_generation", 2, 1, 20, 2^16; discrete=false, get_data_from = "", conditional = true)

#generation_experiment("continuous_production_generation", 1, 2, 2, L, 2^16; seed=120, discrete=false, get_data_from = "", conditional = true)

version = 1

name = "discretely_generated_continuous_prediction"

data_gen_params = OrderedDict(:L => L, :J => J, :g => g, :t => [0.1], :num_samples=>(2^x for x=9:16))
exp_mode = OrderedDict(:load => "generated_production_generation")
hyper_params = OrderedDict(:gaussian_num => 32)

simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)

