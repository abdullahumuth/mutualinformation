using Pkg 
Pkg.activate(".")

include("./transformer.jl")
include("./create_dataset.jl")
include("./save_results.jl")
include("./sample_from_nn.jl")
using DataFrames
using Distributions
using HDF5
using Base.Iterators: product
using OrderedCollections
using JSON3
using Dates


function simple_experiment(name, version, data_gen_params; exp_modes_params=OrderedDict(), hyper_parameters=OrderedDict())
    name = "$(name)_v$(version)"

    for data_gen_param_list in product(values(data_gen_params)...), exp_modes_param_list in product(values(exp_modes_params)...), hyper_param_list in product(values(hyper_parameters)...)
        
        data_gen_param_dict = NamedTuple(zip(keys(data_gen_params), data_gen_param_list))
        exp_modes_param_dict = NamedTuple(zip(keys(exp_modes_params), exp_modes_param_list))
        hyper_param_dict = NamedTuple(zip(keys(hyper_parameters), hyper_param_list))

        data_gen_param_dict, exp_modes_param_dict, hyper_param_dict = special_update(data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
        println("Running experiment with parameters: ", name, " ", data_gen_param_dict, " ", exp_modes_param_dict, " ", hyper_param_dict)

        input = create_dataset(data_gen_param_dict...; exp_modes_param_dict...)
        entropy, conditional_entropy = mutualinformation(input...; hyper_param_dict...)

        save_results(entropy, conditional_entropy, name, data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
    end
end

# To avoid mismatch between "gaussian_num" and "discrete" parameters, I added a special update function to update the experiment modes parameter dictionary.

function special_update(data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
    gaussian_num = get(Dict(pairs(hyper_param_dict)), :gaussian_num, 0)
    if gaussian_num != 0 
        exp_modes_param_dict = merge(exp_modes_param_dict, (discrete = false,))
    else
        exp_modes_param_dict = merge(exp_modes_param_dict, (discrete = true,))
    end

    return data_gen_param_dict, exp_modes_param_dict, hyper_param_dict
end

                

# The new keyword is transfer learning the embedding layer from the entropy model to the conditional entropy model.
# Based on my experiments, that method did not improve the performance much.

function mutualinformation(X,Y; gaussian_num = 0, new = false, kwargs...)
    a_input_dim = size(X)[1]
    b_input_dim = size(Y)[1]
    model = GeneralTransformer(a_input_dim = a_input_dim, gaussian_num = gaussian_num)
    a = train(model, X; kwargs...)
    conditional_model = GeneralTransformer(a_input_dim = a_input_dim, b_input_dim = b_input_dim, gaussian_num = gaussian_num)
    if new
        first_mha = deepcopy(Flux.params(model.a_embed))
        Flux.loadparams!(conditional_model.a_embed, first_mha)
    end
    b = train(conditional_model, X, Y; kwargs...) 
    return a, b
end



