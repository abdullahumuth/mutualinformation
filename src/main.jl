using Pkg 
Pkg.activate(".")

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
    new::Bool
end

function experiment(name, version, L, J, g, t, num_samples; new = false)
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
    return experiment(name, L, J, g, t, num_samples, new)
end

function mkdir_safe(path)
    try mkdir(path) catch e @warn "Probably file already exists: " * e.msg end
end

function (exp::experiment)()
    mkdir_safe("data/outputs/$(exp.name)")
    mkdir_safe("data/outputs/$(exp.name)/models")
    mkdir_safe("data/outputs/$(exp.name)/losses")
    mkdir_safe("data/outputs/$(exp.name)/plots")
    mkdir_safe("data/outputs/$(exp.name)/results")

    for L in exp.L, J in exp.J, g in exp.g, t in exp.t, num_samples in exp.num_samples
        c = exp.name, L, J, g, t, num_samples
    #try
        entropy, conditional_entropy = mutualinformation(c[2:end]..., new = exp.new)
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
    #catch e
    #    @warn "Failed to compute mutual information: " * e.msg
    #end

    end
end


function mutualinformation(L, J, g, t, num_samples; new = false)
    psi = read_wavefunction(L, J, g, t)
    x_proto = stack(gen_samples(psi, num_samples, L), dims = 2) |> todevice
    x = zeros((2, size(x_proto)...))
    x[1, :, :] .= (x_proto .== 1)
    x[2, :, :] .= (x_proto .== -1)
    x = Int.(x)

    psi_vectorized = cat(transpose(real(psi)), transpose(imag(psi)), dims = 1) |> todevice
    y = mapslices(x_proto, dims = 1) do xi
        index = parse(Int, join(string.(Int.(xi .== 1))), base=2)
        return Float32.(psi_vectorized[:,index+1])
    end
    y = reshape(y, (1, size(y)...))


    model = GeneralTransformer()
    a = train(model, x; progress_bar=false, auto_stop=false)
    conditional_model = GeneralTransformer(b_input_dim = 1)
    if new
        first_mha = deepcopy(Flux.params(model.decoder.blocks.:(1).attention))
        Flux.loadparams!(conditional_model.decoder.blocks.:(1).attention, first_mha)
    end
    b = train(conditional_model, x, y; progress_bar=false, auto_stop=false) 
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

function output_csv(a,b,c...)
    result = DataFrame(mutual_information = a[1]-b[1], entropy = a[1], conditional_entropy = b[1])
    CSV.write("data/outputs/$(c[1])/results/" * name_files(c...) * ".csv", result)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("data/outputs/$(c[1])/losses/entropy_" * name_files(c...) * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("data/outputs/$(c[1])/losses/conditional_entropy_" * name_files(c...) * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,c...)
    savefig(plot_models(a,b), "data/outputs/$(c[1])/plots/" * name_files(c...) * ".png")
end



L = 12
J = -1
g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 0.1   # can be anything from collect(0:0.001:1)

#sample_experiment = experiment("sample_convergence_t0t1", 1, L, J, g, 0.1:0.9:1.0, 1000:2000:51000)
#
#transfer_sample_experiment = experiment("transfer_sample_convergence_t0t1", 1, L, J, g, 0.1:0.9:1.0, 1000:2000:51000, new = true)
#
#time_evolve_experiment = experiment("time_evolve", 1, 10:2:18, J, -2.0:1.0:-1.0, 0.0:0.1:1.0, 10000:10000:20000)
#
#sample_experiment()
#transfer_sample_experiment()
#time_evolve_experiment()
# display(CUDA.device())
# test = experiment("testbrooo", 2, 12, -1, -1.0, 0.0, 32, new=false)
# test()

sample_experiment = experiment("sample_convergence_t0t1", 2, L, J, g, 0.1:0.9:1.0, 31000:5000:51000)
sample_experiment()