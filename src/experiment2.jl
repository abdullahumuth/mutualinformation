include("./main.jl")

name = "noise0.00001_discrete_sample"
version = 1

# experiment1 = experiment(L, J, g, 0.1, (2^x for x=5:6))
# experiment1(name, version)

data_gen_params = OrderedDict(:L => [20, 12], :J => J, :g => g, :t => [0.1, 0.5, 1.0], :num_samples=>(2^x for x=5:16))
exp_mode = OrderedDict(:discrete => true, :noise => [0.00001, 0.0001])
hyper_params = OrderedDict()

simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)

