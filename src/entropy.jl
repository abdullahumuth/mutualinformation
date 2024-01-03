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
        SinCosPositionEmbed(hidden_dim),
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

function gaussian_mix(ps, x)
    weights = softmax(ps[:,1])
    gvals = weights .* exp.( -0.5 * ((x .- ps[:,2]) ./ ps[:,3]).^2 ) ./ (sqrt.(2*pi*(ps[:,3].^2)))
    
    return sum(gvals)
end

function (m::GaussianMixTransformer)(input)
    
    x = hcat([0.0], input)

    h = encoder_forward(m, x[:,1:end-1])[:hidden_state]

    gm_params = m.final_dense(h)

    gm_params = reshape(gm_params, (:,3,size(input)[end]))
    
    probs = gaussian_mix.(eachslice(gm_params, dims=3), eachcol(input))
    
    return prod(probs)

end

function (m::GaussianMixTransformer)(x, y)
    
    input = hcat(y,x)

    h = encoder_forward(m, input)[:hidden_state]

    gm_params = m.final_dense(h)[:,size(y)[end]:end-1,:]

    gm_params = reshape(gm_params, (:,3,size(x)[end]))
    
    probs = gaussian_mix.(eachslice(gm_params, dims=3), eachcol(x))
    
    return prod(probs)

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
    X = X ./ std(X, dims=2)

    X_train = X[:,1:num_train]
    X_test = X[:,num_train+1:num_train+num_test]
    X_valid = X[:,num_train+num_test+1:end]

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
                -sum(log.(m.(transpose.(eachcol(x)))))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[2])
        push!(test_losses, -mean(log.(model.(transpose.(eachcol(X_test))))))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                min_epoch = epoch
            end
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)
    H_X = -mean(log.(model.(transpose.(eachcol(X_valid)))))

    return H_X, (train_losses = losses, test_losses = test_losses, min_epoch = min_epoch)

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

    X_train = X[:,1:num_train]
    Y_train = Y[:,1:num_train]
    X_test = X[:,num_train+1:num_train+num_test]
    Y_test = Y[:,num_train+1:num_train+num_test]
    X_valid = X[:,num_train+num_test+1:end]
    Y_valid = Y[:,num_train+num_test+1:end]

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
                -sum(log.(m.(transpose.(eachcol(x)), transpose.(eachcol(y)))))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[2])
        push!(test_losses, -mean(log.(model.(transpose.(eachcol(X_test)), transpose.(eachcol(Y_test))))))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                min_epoch = epoch
            end
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)
    H_XY = -mean(log.(model.(transpose.(eachcol(X_valid)), transpose.(eachcol(Y_valid)))))

    return H_XY, (train_losses = losses, test_losses = test_losses, min_epoch = min_epoch)

end

end