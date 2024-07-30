
using Random
using Statistics
using Flux
using Flux.Functors
using Transformers
using Transformers.Layers
using NeuralAttentionlib
using ProgressMeter
import CUDA

padding_constant = 0

struct GeneralTransformer{P <: Transformers.Layers.AbstractEmbedding}
    general_embed::Dense
    pos_embed::P
    final_dense::Dense
    decoder::Transformers.Layers.Transformer
    encoder::Union{Nothing, Transformers.Layers.Transformer}
    attention_mask
end

function GeneralTransformer(;
    embedding_dim::Integer = 64, 
    head_num::Integer = 4, 
    head_dim::Integer = 8, 
    layer_num::Integer = 2,
    ffn_dim::Integer = 128,
    conditional::Bool = false)
    if conditional
        decoder = Transformer(PreNormTransformerDecoderBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
        encoder = Transformer(PreNormTransformerBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
    else
        decoder = Transformer(PreNormTransformerBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
        encoder = nothing
    end

    return GeneralTransformer(
        Dense(1 => embedding_dim) |> todevice,
        SinCosPositionEmbed(embedding_dim) |> todevice,
        Dense(embedding_dim=>1, sigmoid) |> todevice,
        decoder |> todevice,
        encoder |> todevice,
        NeuralAttentionlib.CausalMask())
end


function embedding(m::GeneralTransformer, x)
    x = reshape(x, (1, size(x)...)) |> todevice
    we = m.general_embed(x)
    pe = m.pos_embed(we)
    return we .+ pe
end

function encoder_forward(m::GeneralTransformer, y)
    y = reshape(y, (1, size(y)...)) |> todevice
    e = embedding(m, y)
    t = m.encoder(e, nothing)
    return t.hidden_state
end

function cross_attend(m::GeneralTransformer, input, encoder_output)
    e = embedding(m, input)
    t = m.decoder(e, encoder_output, m.attention_mask, nothing)
    return t.hidden_state
end

function decoder_forward(m::GeneralTransformer, input)
    e = embedding(m, input)
    t = m.decoder(e, m.attention_mask)
    #println("shape of hidden state: ", size(t.hidden_state))
    return t.hidden_state
end


# we pad to be able to calculate the unconditional probability
# with the final dense layer we calculate the probability of the next token == 1 (with 1d sigmoid input)
# after the final dense layer, we remove the prediction bit
# then we calculate the cross entropy (I just want to calculate p(x), that's why I choose h when x = 1 and 1-h when x = 0)
# we will need the sum of the logarithms of the probabilities, that would be the -log likelihood.
function (m::GeneralTransformer)(x)
    #println("shape of x: ", size(x))
    padded_x = pad_constant(x, (1,0,0,0), padding_constant) 
    h = decoder_forward(m, padded_x)
    h = m.final_dense(h)
    #println("shape after the final dense h: ", size(h))
    h = h[:,1:end-1,:]
    h = reshape(h, (size(h)[2:3]...))
    #println("shape of h: ", size(h))
    h = h.*(x .== 1) + (1 .- h).*(x .== -1)
    h = log2.(h)
    h = -sum(h, dims=1)
    return h
end


#conditional probability
function (m::GeneralTransformer)(x, y)
    
    padded_x = pad_constant(x, (1,0,0,0), padding_constant)
    encoded = encoder_forward(m,y)
    h = cross_attend(m, padded_x, encoded)
    h = m.final_dense(h)
    h = h[:,1:end-1,:]
    h = reshape(h, (size(h)[2:3]...))
    h = h.*(x .== 1) + (1 .- h).*(x .== -1)
    h = log2.(h) 
    h = -sum(h, dims=1)
    return h
end

Flux.@functor GeneralTransformer


function compute_entropy(model, X;
    learning_rate = 1e-4,
    num_steps = 500,
    batch_size = 128,
    test_fraction = 0.2,
    validation_fraction = 0.2,
    seed = 0,
    progress_bar = false)

    Random.seed!(seed)

    # organize data
    num_samples = size(X)[2]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid


    X_train = X[:,1:num_train]
    X_test = X[:,num_train+1:num_train+num_test]
    X_valid = X[:,num_train+num_test+1:end]

    # training
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
                sum(m(x))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[end])
        push!(test_losses, mean(model(X_test)))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                println("Minimum achieved at epoch ", epoch)
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
    H_X = mean(model(X_valid))
    return H_X, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch)

end

function compute_conditional_entropy(model, X, Y;
    learning_rate = 1e-4,
    num_steps = 500,
    batch_size = 128,
    test_fraction = 0.1,
    validation_fraction = 0.1,
    seed = 0,
    progress_bar = false,
    auto_stop = true)

    Random.seed!(seed)

    # organize data
    num_samples = size(X)[2]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid

    # normalize to unit variance

    X_train = X[:,1:num_train]
    X_test = X[:,num_train+1:num_train+num_test]
    X_valid = X[:,num_train+num_test+1:end]

    Y_train = Y[:,1:num_train]
    Y_test = Y[:,num_train+1:num_train+num_test]
    Y_valid = Y[:,num_train+num_test+1:end]


    # training
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
        tmp_params = nothing
        num_batches = 0
        for (x,y) in loader
            num_batches += 1
            loss, grads = Flux.withgradient(model) do m
                sum(m(x, y))
            end

            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(X_train)[end])
        push!(test_losses, mean(model(X_test, Y_test)))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                println("Minimum achieved at epoch ", epoch)
                min_epoch = epoch
            end

            if auto_stop && max(losses[end-min(100, size(losses)[1]-1):end]...) < min(test_losses...)
                break
            end
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)

    H_XY = mean(model(X_valid, Y_valid))

    return H_XY, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch, net=model)

end

