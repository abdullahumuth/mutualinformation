using Plots
using BSON
using CSV
using DataFrames
using Distributions
using HDF5
using OrderedCollections
using JSON3
using Dates



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

