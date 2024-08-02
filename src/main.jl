include("./transformer.jl")
include("./gen_samples.jl")
include("./read_wvfct.jl")
using Plots
using BSON
using CSV
using DataFrames

struct experiment
    name::String
    L::AbstractRange{}
    J::AbstractRange{}
    g::AbstractRange{}
    t::AbstractRange{}
    num_samples::AbstractRange{}
end

function experiment(name, version, L, J, g, t, num_samples)
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
    name = "$(name)_v$(version)"
    return experiment(name, L, J, g, t, num_samples)
end

function (exp::experiment)()
    last_pwd = pwd()
    try
        mkdir("./data/outputs/$(e.name)")
    catch e
        println("Failed to create directory: ", e)
    end

    try
        cd("./data/outputs/$(e.name)")
    catch e
        println("Failed to change directory: ", e)
    end

    for L in exp.L, J in exp.J, g in exp.g, t in exp.t, num_samples in exp.num_samples
        c = exp.name, L, J, g, t, num_samples
    try
        entropy, conditional_entropy = mutualinformation(c[2:end]...)   
        try
            save_models(entropy, conditional_entropy, c...)
        catch e
            @warn "Failed to save models: " * e.msg
        end
        try
            output_csv(entropy, conditional_entropy, c...)
        catch e
            @warn "Failed to save csv: " * e.msg
        end
        try
            save_plots(entropy, conditional_entropy, c...)
        catch e
            @warn "Failed to save plots: " * e.msg
        end
    catch e
        @warn "Failed to compute mutual information: " * e.msg
        end
    end
    cd(last_pwd)
end


function mutualinformation(L, J, g, t, num_samples)
    psi = read_wavefunction(L, J, g, t)
    x = stack(gen_samples(psi, num_samples, L), dims = 2) |> todevice
    psi_vectorized = cat(transpose(real(psi)), transpose(imag(psi)), dims = 1) |> todevice
    y = mapslices(x, dims = 1) do xi
        index = parse(Int, join(string.(Int.(xi .== 1))), base=2)
        return Float32.(psi_vectorized[:,index+1])
    end
    model = GeneralTransformer()
    a = train(model, x; progress_bar=true, auto_stop=false)
    conditional_model = GeneralTransformer(conditional=true)
    b = train(conditional_model, x, y; progress_bar=true, auto_stop=false) 
    return a, b
end

function plot_models(a, b)
    p = plot(a[2].train_losses, label="train", title="Entropy= $(a[1])")
    plot!(a[2].test_losses, label="test")
    annotate!(a[2].min_epoch, a[2].test_losses[a[2].min_epoch], text("Minimum at $(a[2].min_epoch)", 12, :left))
    p2 = plot(b[2].train_losses, label="train", title="Conditional Entropy= $(b[1])")
    annotate!(b[2].min_epoch, b[2].test_losses[b[2].min_epoch], text("Minimum at $(b[2].min_epoch)", 12, :left))
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
        bson("model_" * name_files(c...) * ".bson", entropy = entropy_model, conditional_entropy = conditional_entropy_model)
    end
end

function output_csv(a,b,c...)
    result = DataFrame(mutual_information = a[1]-b[1], entropy = a[1], conditional_entropy = b[1])
    CSV.write("result_" * name_files(c...) * ".csv", result)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("entropy_loss_" * name_files(c...) * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("conditional_entropy_loss_" * name_files(c...) * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,c...)
    savefig(plot_models(a,b), "plot_" * name_files(c...) * ".png")
end



L = 12
J = -1
g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 0.0   # can be anything from collect(0:0.001:1)
num_samples = 30
version = 1

e = experiment("test", version, L, J, g, t, num_samples)
e()


# For the causal self-attention layer of conditional entropy estimator, 
# I may use the same parameters from the entropy estimator and see how it performs.

