include("./main.jl")
include("plot_from_json.jl")
name = "learning_rate_tests"
version = 1

# experiment1 = experiment(L, J, g, 0.1, (2^x for x=5:6))
# experiment1(name, version)

data_gen_params = OrderedDict(:L => [20, 12], :J => J, :g => g, :t => [0.1, 0.5, 0.9], :num_samples=>(2^x for x=5:16))
exp_mode = OrderedDict()
hyper_params = OrderedDict(:gaussian_num => [0,32], :learning_rate => (10.0^x for x=-5:-1))

simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)
