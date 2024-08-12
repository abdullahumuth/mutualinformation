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

function (exp::experiment)(name="nameless_exp", version=1;kwargs...)
    name = "$(name)_v$(version)"

    mkdir_safe("data/outputs/$(name)")
    mkdir_safe("data/outputs/$(name)/models")
    mkdir_safe("data/outputs/$(name)/losses")
    mkdir_safe("data/outputs/$(name)/plots")
    mkdir_safe("data/outputs/$(name)/results")

    for L in exp.L, J in exp.J, g in exp.g, t in exp.t, num_samples in exp.num_samples
        c = name, L, J, g, t, num_samples
    #try
        entropy, conditional_entropy = mutualinformation(inputhandler(c[2:end]...)...; kwargs...)
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

function inputhandler(L,J,g,t,num_samples)
    #psi = read_wavefunction(L, J, g, t)
    #dist = Categorical(abs2.(psi))

    dist = DiscreteUniform(1, 2^L)

    indices = rand(dist, num_samples)
    f(x) = digits(x, base=2, pad = L)|> reverse
    x_proto = stack(map(f, indices .- 1))

    x = zeros((2, size(x_proto)...))
    x[1, :, :] .= x_proto
    x[2, :, :] .= 1 .- x_proto
    x = Int.(x) |> gpu

    y = fake_y(L; unit=1, offset=10, partition=2)[:, indices]

    #y = stack(map(x -> [real(psi[x]), imag(psi[x])], indices))

    y = reshape(y, (1, size(y)...)) |> gpu
    return x, y
end

function mutualinformation(X,Y; new = false, kwargs...)
    a_input_dim = size(X)[1]
    b_input_dim = size(Y)[1]
    model = GeneralTransformer(a_input_dim = a_input_dim)
    a = train(model, X; kwargs...)
    conditional_model = GeneralTransformer(a_input_dim = a_input_dim, b_input_dim = b_input_dim)
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



L = 20
J = -1
g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 0.1   # can be anything from collect(0:0.001:1)

# test = experiment(L, J, g, 0.1, 100)
# test("test", 1, new = true)


# sample_experiment = experiment(L, J, g, 0.1:0.9:1.0, (2^x for x=9:16))
# sample_experiment("sample_convergence_large_batch128", 1; )
# sample_experiment("transfer_sample_convergence_large_batch128", 1; new = true)
# sample_experiment("sample_convergence_large_batch256", 1; batch_size = 256)
# sample_experiment("sample_convergence_large_batch512", 1; batch_size = 512)
# sample_experiment("sample_convergence_large_batch1024", 1; batch_size = 1024)
# sample_experiment("transfer_sample_convergence_large_batch1024", 1; batch_size = 1024, new = true)

# time_evolve_experiment = experiment(20, J, g, 0.0:0.05:1.0, 10000)
# 
# time_evolve_experiment("time_evolve_convergence_large_batch256", 1, batch_size = 256)
# time_evolve_experiment("transfer_time_evolve_convergence_large_batch256", 1; new = true, batch_size = 256)



moment_of_truth = experiment(L, J, g, 0.0, (2^x for x=9:16))
moment_of_truth("moment_of_truth_batch128_2partition", 1; batch_size = 128)

