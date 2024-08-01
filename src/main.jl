include("./transformer.jl")
include("./gen_samples.jl")
include("./read_wvfct.jl")
using Plots
using BSON
using CSV
using DataFrames

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
    p = plot(a[2].train_losses, label="train")
    plot!(a[2].test_losses, label="test")
    annotate!(length(a[2].train_losses)/4, a[2].test_losses[end/4], text("min epoch: $(a[2].min_epoch)", 12, :left))
    annotate!(length(a[2].train_losses)/2, a[2].test_losses[end/2], text("final entropy: $(a[1])", 12, :left))
    p2 = plot(b[2].train_losses, label="train")
    plot!(b[2].test_losses, label="test")
    annotate!(length(b[2].train_losses)/4, b[2].test_losses[end/4], text("min epoch: $(b[2].min_epoch)", 12, :left))
    annotate!(length(b[2].train_losses)/2, b[2].test_losses[end/2], text("final conditional entropy: $(b[1])", 12, :left))
    return plot(p,p2, layout = (2,1))
end

function name_files(L, J, g, t, num_samples, VERSION)
    return "_L=$(L)_J=$(J)_g=$(g)_t=$(t)_num_samples=$(num_samples)_v$(VERSION)"
end

function save_models(a,b,c)
    L, J, g, t, num_samples, VERSION = c
    let 
        entropy_model = cpu(a[2].net)
        conditional_entropy_model = cpu(b[2].net)
        write("/data/outputs/models/models" * name_files(L, J, g, t, num_samples, VERSION) * ".bson", " ")
        bson("/data/outputs/models/models" * name_files(L, J, g, t, num_samples, VERSION) * ".bson", entropy = entropy_model, conditional_entropy = conditional_entropy_model)
    end
end

function output_csv(a,b,c)
    L, J, g, t, num_samples, VERSION = c
    result = DataFrame(mutual_information = a[1]-b[1], entropy = a[1], conditional_entropy = b[1])
    CSV.write("/data/outputs/results/result" * name_files(L, J, g, t, num_samples, VERSION) * ".csv", result)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("/data/outputs/losses/entropy_loss" * name_files(L, J, g, t, num_samples, VERSION) * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("/data/outputs/losses/conditional_entropy_loss" * name_files(L, J, g, t, num_samples, VERSION) * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,c)
    L, J, g, t, num_samples, VERSION = c
    savefig(plot_models(a,b), "/data/outputs/plots/plot" * name_files(L, J, g, t, num_samples, VERSION) * ".png")
end



L = 12
J = -1
g = -1.0 # can be anything from [-0.5,-1.0,-2.0]
t = 0.0   # can be anything from collect(0:0.001:1)
num_samples = 30
version = 1

c = L, J, g, t, num_samples, version

entropy, conditional_entropy = mutualinformation(L, J, g, t, num_samples)
#save_models(entropy, conditional_entropy, c)
output_csv(entropy, conditional_entropy, c)
save_plots(entropy, conditional_entropy, c)


# For the causal self-attention layer of conditional entropy estimator, 
# I may use the same parameters from the entropy estimator and see how it performs.

