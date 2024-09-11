include("./main.jl")
include("plot_from_json.jl")
name = "newjson"
version = 1

L = 20
J = -1
g = -1.0
t = 0.1


# experiment1 = experiment(L, J, g, 0.1, (2^x for x=5:6))
# experiment1(name, version)

# data_gen_params = OrderedDict(:L => [20, 12], :J => J, :g => g, :t => [0.1, 0.5, 0.9], :num_samples=>(2^x for x=6:16))
# exp_mode = OrderedDict(:discrete => false, :noise => [0.00001, 0.0001, 0.001, 0.01, 0.1])
# hyper_params = OrderedDict(:gaussian_num => 32)

data_gen_params = OrderedDict(:L => 12, :J => J, :g => g, :t => 0.1, :num_samples=>(2^x for x=5:6))
exp_mode = OrderedDict(:noise => [0.01, 0.1])
hyper_params = OrderedDict(:max_epochs => 30)



simple_experiment(name, version, data_gen_params; exp_modes_params = exp_mode, hyper_parameters = hyper_params)


