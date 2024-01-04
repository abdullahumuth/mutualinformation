module Entropy

export compute_entropy,
        compute_conditional_entropy

using Random
using Statistics
using Flux
using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using NeuralAttentionlib
using ProgressMeter


struct GaussianMixTransformer{P <: Transformers.Layers.AbstractEmbedding, T <: Transformers.Layers.Transformer, A}
    word_embed::Dense
    pos_embed::P
    final_dense::Dense
    trf::T
    attention_mask::A
end

function GaussianMixTransformer(
    hidden_dim::Integer = 64, 
    head_num::Integer = 4, 
    head_dim::Integer = 8, 
    layer_num::Integer = 2,
    ffn_dim::Integer = 128,
    gaussian_num::Integer = 64)

    return GaussianMixTransformer(
        Dense(1 => hidden_dim) |> gpu,
        SinCosPositionEmbed(hidden_dim) |> todevice,
        Dense(hidden_dim=>3*gaussian_num) |> gpu,
        Transformer(TransformerBlock, layer_num, head_num, hidden_dim, head_dim, ffn_dim) |> todevice,
        NeuralAttentionlib.CausalMask()
    )
end


function embedding(m::GaussianMixTransformer, x)
    we = m.word_embed(x)
    pe = m.pos_embed(we)
    return we .+ pe
end

function encoder_forward(m::GaussianMixTransformer, input)
    e = embedding(m, input)
    t = m.trf(e, m.attention_mask)
    return t
end

function weighted_gaussian(x, weight, μ, σ)
    return weight * exp( -0.5 * ((x - μ) / σ)^2 ) / (sqrt(2*pi*(σ^2)))
end

function gaussian_mix(ps, x)

    weights = softmax(ps[:,1,:,:])
    gvals = weighted_gaussian.(x, weights, ps[:,2,:,:], ps[:,3,:,:])
    
    return sum(gvals, dims=1)

end

function (m::GaussianMixTransformer)(input)

    input = reshape(input, (size(input)[1:2]..., :))
    
    # Zero padding for unconditional first probability
    x = pad_zeros(input, (0,0,1,0,0,0))

    h = encoder_forward(m, x[:,1:end-1, :])[:hidden_state]

    gm_params = m.final_dense(h)

    gm_params = reshape(gm_params, (:, 3, size(input)[end-1:end]...))
    
    probs = gaussian_mix(gm_params, input)

    return reshape(prod(probs, dims=2), (:))

end

function (m::GaussianMixTransformer)(x, y)

    x = reshape(x, (size(x)[1:2]..., :))
    y = reshape(y, (size(y)[1:2]..., :))
    
    input = cat(y,x; dims=2)

    h = encoder_forward(m, input)[:hidden_state]

    gm_params = m.final_dense(h)[:,size(y)[2]:end-1,:]

    gm_params = reshape(gm_params, (:,3,size(x)[end-1:end]...))

    probs = gaussian_mix(gm_params, x)
    
    return reshape(prod(probs, dims=2), (:))

end

Flux.@functor GaussianMixTransformer

function compute_entropy(X;
    learning_rate = 1e-4,
    num_steps = 500,
    batch_size = 128,
    test_fraction = 0.1,
    validation_fraction = 0.1,
    seed = 0,
    progress_bar = false)

    Random.seed!(seed)

    # organize data
    num_samples = size(X)[2]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid

    # normalize to unit variance
    X = X ./ std(X, dims=2) |> gpu

    if length(size(X)) < 3
        X = reshape(X, (1, size(X)...))
    end
    X_train = X[:,:,1:num_train]
    X_test = X[:,:,num_train+1:num_train+num_test]
    X_valid = X[:,:,num_train+num_test+1:end]

    # training
    model = GaussianMixTransformer()
    optimal_params = deepcopy(Flux.params(model))

    optim = Flux.setup(Flux.Adam(learning_rate), model)
    loader = Flux.DataLoader((X_train,), batchsize=batch_size, shuffle=true);

    losses = []
    test_losses = []
    min_epoch = 0

    epochs = 1:num_steps
    progress = nothing
    if progress_bar
        progress = Progress(num_steps; dt=1.0)
    end
    for epoch in epochs
        accumulated_loss = 0
        for (x,) in loader
            loss, grads = Flux.withgradient(model) do m
                -sum(log.(m(x)))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[end])
        push!(test_losses, -mean(log.(model(X_test))))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                min_epoch = epoch
            end
        end

        if isnan(test_losses[end])
            break
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)
    H_X = -mean(log.(model(X_valid)))

    return H_X, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch)

end

function compute_conditional_entropy(X, Y;
    learning_rate = 1e-4,
    num_steps = 500,
    batch_size = 128,
    test_fraction = 0.1,
    validation_fraction = 0.1,
    seed = 0,
    progress_bar = false)

    Random.seed!(seed)

    # organize data
    num_samples = size(X)[2]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid

    # normalize to unit variance
    X = X ./ std(X, dims=2) |> gpu
    Y = Y ./ std(Y, dims=2) |> gpu

    if length(size(X)) < 3
        X = reshape(X, (1, size(X)...))
    end
    if length(size(Y)) < 3
        Y = reshape(Y, (1, size(Y)...))
    end

    X_train = X[:,:,1:num_train]
    Y_train = Y[:,:,1:num_train]
    X_test = X[:,:,num_train+1:num_train+num_test]
    Y_test = Y[:,:,num_train+1:num_train+num_test]
    X_valid = X[:,:,num_train+num_test+1:end]
    Y_valid = Y[:,:,num_train+num_test+1:end]

    # training
    model = GaussianMixTransformer()
    optimal_params = deepcopy(Flux.params(model))

    optim = Flux.setup(Flux.Adam(learning_rate), model)
    loader = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true);

    losses = []
    test_losses = []
    min_epoch = 0

    epochs = 1:num_steps
    progress = nothing
    if progress_bar
        progress = Progress(num_steps; dt=1.0)
    end
    for epoch in epochs
        accumulated_loss = 0
        for (x,y) in loader
            loss, grads = Flux.withgradient(model) do m
                -sum(log.(m(x, y)))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[end])
        push!(test_losses, -mean(log.(model(X_test, Y_test))))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                min_epoch = epoch
            end
        end

        if isnan(test_losses[end])
            break
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)
    H_XY = -mean(log.(model(X_valid, Y_valid)))

    return H_XY, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch)

end

end
