using Pkg 
Pkg.activate(".")

include("./transformer.jl")
include("./read_npy_wavefunctions.jl")
include("./read_wvfct.jl")
using Plots
using BSON
using CSV
using DataFrames
using Distributions
using HDF5
using Base.Iterators: product
using OrderedCollections
using JSON3
using Dates

function mkdir_safe(path)
    try mkdir(path) catch e @warn "Probably file already exists: " * e.msg end
end

function simple_experiment(name, version, data_gen_params; exp_modes_params=OrderedDict(), hyper_parameters=OrderedDict())
    name = "$(name)_v$(version)"

    mkdir_safe("data/outputs/$(name)")
    mkdir_safe("data/outputs/$(name)/models")
    mkdir_safe("data/outputs/$(name)/losses")
    mkdir_safe("data/outputs/$(name)/plots")
    mkdir_safe("data/outputs/$(name)/results")

    jsons = Dict()
    try
        output_json(jsons, name)
    catch e
        if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
        @warn "Failed to create JSON " * str
    end
    for data_gen_param_list in product(values(data_gen_params)...), exp_modes_param_list in product(values(exp_modes_params)...), hyper_param_list in product(values(hyper_parameters)...)
        
        data_gen_param_dict = NamedTuple(zip(keys(data_gen_params), data_gen_param_list))
        exp_modes_param_dict = NamedTuple(zip(keys(exp_modes_params), exp_modes_param_list))
        hyper_param_dict = NamedTuple(zip(keys(hyper_parameters), hyper_param_list))

        # Safe access to gaussian_num with a default value of 0
        gaussian_num = get(Dict(pairs(hyper_param_dict)), :gaussian_num, 0)

        # Conditional modification of exp_modes_param_dict
        if gaussian_num != 0 
            exp_modes_param_dict = merge(exp_modes_param_dict, (discrete = false,))
        else
            exp_modes_param_dict = merge(exp_modes_param_dict, (discrete = true,))
        end

        println("Running experiment with parameters: ", name, " ", data_gen_param_dict, " ", exp_modes_param_dict, " ", hyper_param_dict)


        input = inputhandler(data_gen_param_dict...; exp_modes_param_dict...)
        entropy, conditional_entropy = mutualinformation(input...; hyper_param_dict...)


        file_name = name_files(merge(data_gen_param_dict, exp_modes_param_dict, hyper_param_dict))
        try
            save_models(entropy, conditional_entropy, name, file_name)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save models: " * str
        end
        try
           output_csv(entropy, conditional_entropy, name, file_name)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save csv: " * str
        end
        try
            json = read("data/outputs/$name/results/result.json", String)
            experiments = copy(JSON3.read(json))[:experiments]
            experiments[Symbol(file_name)] = output(entropy, conditional_entropy, data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
            output_json(experiments, name)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save JSON " * str
        end
        
        try
            save_plots(entropy, conditional_entropy, name, file_name)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save plots: " * str
        end
    end
end
                


function inputhandler(L,J,g,t,num_samples; noise=0, load = "", discrete=true, uniform = false, unique = false, fake = false, shuffle=false)
    if load != ""
        data = load_generated_data(load, num_samples, discrete)
        return data.data |> gpu
    end
    if L == 20
        psi = read_wavefunction(L, J, g, t)
    else
        psi = read_hdf5_wavefunction(L, J, g, t)
    end
    if unique
        indices = randperm(Xoshiro(303), 2^L)[1:num_samples]
    else
        if uniform
            dist = DiscreteUniform(1, 2^L)
        else
            dist = Categorical(abs2.(psi))
        end
        indices = rand(Xoshiro(303), dist, num_samples)
    end

    f(x) = digits(x, base=2, pad = L)|> reverse
    x_proto = stack(map(f, indices .- 1))

    x = zeros((2, size(x_proto)...))
    x[1, :, :] .= x_proto
    x[2, :, :] .= 1 .- x_proto
    x = Int.(x) |> gpu


    if fake
        y = fake_y(L; unit=1, offset=16, partition=2, shuffle=shuffle)[:, indices]
    else
        y = stack(map(x -> [real(psi[x]), imag(psi[x])], indices))
    end
    noise > 0 && (y .+= randn(Xoshiro(303), size(y)) .* noise)
    y = reshape(y, (1, size(y)...)) |> gpu
    discrete && (return x, y)
    return y, x
end

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

function plot_models(a, b)
    p = plot(a[2].train_losses, label="train", title="Entropy= $(a[1])  (min. $(a[2].min_epoch))")
    plot!(a[2].test_losses, label="test")
    p2 = plot(b[2].train_losses, label="train", title="Conditional Entropy= $(b[1])  (min. $(b[2].min_epoch))")
    plot!(b[2].test_losses, label="test")
    return plot(p,p2, layout = (2,1))
end


function name_files(dict)
    return join(["$(k)=$(v)_" for (k,v) in zip(keys(dict), values(dict))])
end

function save_models(a, b, exp_name, file_name)
    let 
        entropy_model = cpu(a[2].net)
        conditional_entropy_model = cpu(b[2].net)
        bson("data/outputs/$exp_name/models/" * file_name * ".bson", entropy = entropy_model, conditional_entropy = conditional_entropy_model)
    end
end

function output(a, b, params, exp_params, hyper_params)

    # Construct the data structure
    data = Dict(
        "parameters" => params,
        "experiment_modes" => exp_params, 
        "hyperparameters" => hyper_params,
        "results" => Dict(
            "mutual_information" => a[1] - b[1],
            "entropy" => a[1],
            "conditional_entropy" => b[1]
        ),
        "other_results" => Dict(
            "avg_time_entropy" => a[2].avg_time,
            "avg_time_conditional_entropy" => b[2].avg_time
        )
    )
    return data
end


function output_json(data, exp_name)
    # Create the directory if it doesn't exist
    mkpath("data/outputs/$exp_name/results")

    # Generate the filename
    filename = "data/outputs/$exp_name/results/result.json"

    


    # Write to JSON file
    open(filename, "w") do io
        JSON3.pretty(io, OrderedDict("experiments" => data,
        "metadata" => OrderedDict(
            "date" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
            "number_of_experiments" => length(data)
        )))
    end
end
    


function output_csv(a,b, exp_name, file_name)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("data/outputs/$exp_name/losses/entropy_" * file_name * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("data/outputs/$exp_name/losses/conditional_entropy_" * file_name * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,exp_name, file_name)
    savefig(plot_models(a,b), "data/outputs/$exp_name/plots/" * file_name * ".png")
end

function generator(name, model_output_dim = 2, model_output_seq_len = 20, initially_generated_dim = 1, initially_generated_seq_len=2, num_samples=10000, seed=313; conditional=true, discrete=true)
    rng = Xoshiro(seed)
    if discrete
        if conditional
            m = GeneralTransformer(a_input_dim = model_output_dim, b_input_dim = initially_generated_dim)
            y = (2 .* rand(rng, (initially_generated_dim, initially_generated_seq_len, num_samples)) .- 1) |> gpu
            x = generate_samples(m, model_output_dim, model_output_seq_len, num_samples, y) |> gpu
            output = (x, y)
        else
            m = GeneralTransformer(a_input_dim = model_output_dim)
            x = generate_samples(m, model_output_dim, model_output_seq_len, num_samples, nothing) |> gpu
            output = (x,)
        end
    else
        if conditional
            m = GeneralTransformer(a_input_dim = model_output_dim, b_input_dim = initially_generated_dim, gaussian_num = 32)
            x_proto = bitrand(rng, (initially_generated_seq_len, num_samples))
            if initially_generated_dim != 2
                throw(ArgumentError("Only 2 initially generated dimensions are supported"))
            end
            x = zeros((initially_generated_dim, size(x_proto)...))
            x[1, :, :] .= x_proto
            x[2, :, :] .= 1 .- x_proto
            x = Int.(x) |> gpu
            y = generate_samples(m, model_output_dim, model_output_seq_len, num_samples, x; discrete = false) |> gpu
            output = (y, x)
        else
            m = GeneralTransformer(a_input_dim = model_output_dim, gaussian_num = 32)
            y = generate_samples(m, model_output_dim, model_output_seq_len, num_samples, nothing; discrete = false) |> gpu
            output = (y,)
        end

    end

    mkdir_safe("data/inputs/$(name)")
    mkdir_safe("data/inputs/$(name)/models")
    mkdir_safe("data/inputs/$(name)/data")
    let 
        model = cpu(m)
        bson("data/inputs/$(name)/models/generated_" * name * ".bson", Dict(:model => model))
    end
    h5open("data/inputs/$(name)/data/generated_" * name * ".h5", "w") do file
        g = create_group(file, "data")
        if length(output) > 1
            if discrete
                g["x"] = cpu(output[1])
                g["y"] = cpu(output[2])
            else
                g["y"] = cpu(output[1])
                g["x"] = cpu(output[2])
            end
        else
            if discrete
                g["x"] = cpu(output[1])
            else
                g["y"] = cpu(output[1])
            end
        end
    end
    return (model = m, data = output)
end

function load_generated_data(name, num_samples, discrete=true; conditional=true)
    file = h5open("data/inputs/$(name)/data/generated_" * name * ".h5", "r")
    data = read(file, "data")
    close(file)
    m = BSON.load("data/inputs/$(name)/models/generated_" * name * ".bson")
    if discrete
        indices = randperm(Xoshiro(303), size(data["x"])[3])[1:num_samples]
        if conditional
            full_data = (data["x"][:,:,indices], data["y"][:,:,indices])
        else
            full_data = (data["x"][:,:,indices],)
        end
    else
        indices = randperm(Xoshiro(303), size(data["y"])[3])[1:num_samples]
        if conditional
            full_data = (data["y"][:,:,indices], data["x"][:,:,indices])
            println("Data loaded")
        else
            full_data = (data["y"][:,:,indices],)
            println("Data loaded")
        end
    end

    return (model = m[:model], data = full_data)
end


function evaluate(model, x, y; discrete = true)
    if isnothing(y)
        return mean(model(x; discrete = discrete))
    end
    return mean(model(x, y; discrete = discrete))
    
end

function generation_experiment(name, model_output_dim = 2, model_output_seq_len = 20, initially_generated_dim = 1, initially_generated_seq_len=2, num_samples = 10000; seed=313, get_data_from = "", discrete=true, conditional = true, kwargs...)
    if get_data_from == ""
        get_data_from = name
        m = generator(name, model_output_dim, model_output_seq_len, initially_generated_dim, initially_generated_seq_len, num_samples, seed; conditional = conditional, discrete = discrete)
    else
        m = load_generated_data(get_data_from, num_samples, discrete; conditional=conditional)
    end
    mkdir_safe("data/inputs/$(get_data_from)")
    mkdir_safe("data/inputs/$(get_data_from)/results")

    gpu_model, gpu_data = m.model |> gpu, m.data |> gpu

    if conditional
        a = evaluate(gpu_model, gpu_data...; discrete = discrete, kwargs...)
        model_name = "conditional_entropy"
    else
        a = evaluate(gpu_model, gpu_data[1], nothing; discrete = discrete, kwargs...)
        model_name = "entropy"
    end
    result = DataFrame(Symbol(model_name) => a)
    CSV.write("data/inputs/$(get_data_from)/results/generated_" * name * ".csv", result)
    println(model_name, " of the data called ", get_data_from, " is ", a)
end


L = 20
J = -1
g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 0.1   # can be anything from collect(0:0.001:1)


