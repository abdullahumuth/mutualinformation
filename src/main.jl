using Pkg 
Pkg.activate(".")

include("./transformer.jl")
include("./read_npy_wavefunctions.jl")
#include("./read_wvfct.jl")
using Plots
using BSON
using CSV
using DataFrames
using Distributions
using HDF5

struct experiment
    L::Union{AbstractRange, Base.Generator}
    J::Union{AbstractRange, Base.Generator}
    g::Union{AbstractRange, Base.Generator}
    t::Union{AbstractRange, Base.Generator}
    num_samples::Union{AbstractRange, Base.Generator}
end

function experiment(L, J, g, t, num_samples)
    if typeof(L) == Int
        L = L:1:L
    end
    if typeof(J) == Int 
        J = J:1:J
    end
    if typeof(num_samples) == Int
        num_samples = num_samples:1:num_samples
    end
    if typeof(g) == Float64
        g = g:1.0:g
    end
    if typeof(t) == Float64
        t = t:1.0:t
    end
    return experiment(L, J, g, t, num_samples)
end

function mkdir_safe(path)
    try mkdir(path) catch e @warn "Probably file already exists: " * e.msg end
end

function (exp::experiment)(name="nameless_exp", version=1; load="", gaussian_num=0, uniform=false, unique=false, fake=false, shuffle=false, kwargs...)
    name = "$(name)_v$(version)"

    mkdir_safe("data/outputs/$(name)")
    mkdir_safe("data/outputs/$(name)/models")
    mkdir_safe("data/outputs/$(name)/losses")
    mkdir_safe("data/outputs/$(name)/plots")
    mkdir_safe("data/outputs/$(name)/results")

    for L in exp.L, J in exp.J, g in exp.g, t in exp.t, num_samples in exp.num_samples
        c = name, L, J, g, t, num_samples
        println("Running experiment with parameters: ", c)
    #try
        entropy, conditional_entropy = mutualinformation(inputhandler(c[2:end]...; load=load, discrete = (gaussian_num==0), uniform=uniform, unique=unique, fake=fake, shuffle=shuffle)...; gaussian_num = gaussian_num, kwargs...)
        
        try
            save_models(entropy, conditional_entropy, c...)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save models: " * str
        end
        try
           output_csv(entropy, conditional_entropy, c...; kwargs...)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save csv: " * str
        end
        try
            save_plots(entropy, conditional_entropy, c...)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to save plots: " * str
        end
    #catch e
    #    if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
    #    @warn "Failed to compute mutual information: " * str
    #end
    end
end

function inputhandler(L,J,g,t,num_samples; load = "", discrete=true, uniform = false, unique = false, fake = false, shuffle=false)
    if load != ""
        data = load_generated_data(load, num_samples)
        discrete && (return data.data)
        throw(ArgumentError("Didn't implement non-discrete data loading yet"))
    end
    psi = read_wavefunction(L, J, g, t)
    if unique
        indices = randperm(MersenneTwister(303), 2^L)[1:num_samples]
    else
        if uniform
            dist = DiscreteUniform(1, 2^L)
        else
            dist = Categorical(abs2.(psi))
        end
        indices = rand(MersenneTwister(303), dist, num_samples)
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

function name_files(name, L, J, g, t, num_samples)
    return "L=$(L)_J=$(J)_g=$(g)_t=$(t)_num_samples=$(num_samples)"
end

function save_models(a,b,c...)
    let 
        entropy_model = cpu(a[2].net)
        conditional_entropy_model = cpu(b[2].net)
        bson("data/outputs/$(c[1])/models/" * name_files(c...) * ".bson", entropy = entropy_model, conditional_entropy = conditional_entropy_model)
    end
end

function output_csv(a,b,c...; kwargs...)
    result = DataFrame(:mutual_information => a[1]-b[1], :entropy => a[1], :conditional_entropy => b[1], :avg_time_entropy => a[2].avg_time, :avg_time_conditional_entropy => b[2].avg_time, kwargs...)
    CSV.write("data/outputs/$(c[1])/results/" * name_files(c...) * ".csv", result)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("data/outputs/$(c[1])/losses/entropy_" * name_files(c...) * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("data/outputs/$(c[1])/losses/conditional_entropy_" * name_files(c...) * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,c...)
    savefig(plot_models(a,b), "data/outputs/$(c[1])/plots/" * name_files(c...) * ".png")
end

function generator(name, input_dim=2, seq_len=20, num_samples=10000;conditional=true)
    if conditional
        m = GeneralTransformer(a_input_dim = 2, b_input_dim = 1)
        y = 2 .* rand(2, num_samples) .- 1
        y = reshape(y, (1, size(y)...)) |> gpu
        x = generate_samples(m, input_dim, seq_len, num_samples, y) |> gpu
        output = (x, y)
    else
        m = GeneralTransformer(a_input_dim = 2)
        x = generate_samples(m, input_dim, seq_len, num_samples, nothing) |> gpu
        output = (x,)
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
        g["x"] = cpu(output[1])
        if length(output) > 1
            g["y"] = cpu(output[2])
        end
    end
    return (model = m, data = output)
end

function load_generated_data(name, num_samples)
    file = h5open("data/inputs/$(name)/data/generated_" * name * ".h5", "r")
    data = read(file, "data")
    close(file)
    println(size(data["x"]))
    println(size(data["y"]))
    m = BSON.load("data/inputs/$(name)/models/generated_" * name * ".bson")
    indices = randperm(MersenneTwister(303), size(data["x"])[3])[1:num_samples]
    return (model = m[:model], data = (data["x"][:,:,indices], data["y"][:,:,indices]))
end


function evaluate(model, x, y; discrete = true)
    if isnothing(y)
        return mean(model(x; discrete = discrete))
    end
    return mean(model(x, y; discrete = discrete))
    
end

function generation_experiment(name, input_dim = 2, seq_len = 20, num_samples = 10000; get_data_from = "", conditional = true, kwargs...)
    if get_data_from == ""
        get_data_from = name
        m = generator(name, input_dim, seq_len, num_samples; conditional = conditional)
    else
        m = load_generated_data(get_data_from, num_samples)
    end
    mkdir_safe("data/inputs/$(get_data_from)")
    mkdir_safe("data/inputs/$(get_data_from)/results")

    if conditional
        a = evaluate(m.model, m.data...; kwargs...)
        model_name = "conditional_entropy"
    else
        a = evaluate(m.model, m.data[1], nothing; kwargs...)
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


