using Random
using Statistics
using Flux
using Flux.Functors
using Transformers
using Transformers.Layers
using NeuralAttentionlib
using ProgressMeter
using CUDA

padding_constant = -1

struct GeneralTransformer{P <: Transformers.Layers.AbstractEmbedding}
    a_embed::Dense
    b_embed::Union{Nothing, Dense}
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
    a_input_dim::Integer = 2,
    b_input_dim::Integer = 0,)
    if b_input_dim != 0
        b_embed = Dense(b_input_dim => embedding_dim)
        decoder = Transformer(PreNormTransformerDecoderBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
        encoder = Transformer(PreNormTransformerBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
    else
        b_embed = nothing
        decoder = Transformer(PreNormTransformerBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
        encoder = nothing
    end

    return GeneralTransformer(
        Dense(a_input_dim => embedding_dim) |> gpu,
        b_embed |> gpu,
        SinCosPositionEmbed(embedding_dim) |> gpu,
        Dense(embedding_dim => a_input_dim) |> gpu,
        decoder |> gpu,
        encoder |> gpu,
        NeuralAttentionlib.CausalMask())
end


function embedding(m::GeneralTransformer, x; encoder=false)
    if encoder
        we = m.b_embed(x)
    else
        we = m.a_embed(x)
    end
    pe = m.pos_embed(we)
    return we .+ pe
end

function encoder_forward(m::GeneralTransformer, y)
    e = embedding(m, y, encoder = true)
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

function helper(x)
    x = log2(exp(1)).* sum(x, dims=2)
    x = reshape(x, (1,size(x)[3]))
    return x
end

# we pad to be able to calculate the unconditional probability
# with the final dense layer we calculate the probability of the next token == 1 (with 1d sigmoid input)
# after the final dense layer, we remove the prediction bit
# then we calculate the cross entropy (I just want to calculate p(x), that's why I choose h when x = 1 and 1-h when x = 0)
# we will need the sum of the logarithms of the probabilities, that would be the -log likelihood.
function (m::GeneralTransformer)(x)
    padded_x = pad_constant(x, (0,0,1,0,0,0), padding_constant) 
    h = decoder_forward(m, padded_x)
    h = m.final_dense(h)
    h = h[:,1:end-1,:]
    h = Flux.logitcrossentropy(h, x, agg = helper)
    return h
end


#conditional probability
function (m::GeneralTransformer)(x, y)
    
    padded_x = pad_constant(x, (0,0,1,0,0,0), padding_constant)
    encoded = encoder_forward(m,y)
    h = cross_attend(m, padded_x, encoded)
    h = m.final_dense(h)
    h = h[:,1:end-1,:]
    h = Flux.logitcrossentropy(h, x, agg = helper)
    return h
end

Flux.@functor GeneralTransformer

function train(model, input...;
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

    num_samples = size(input[1])[3]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid

    
    if length(input) == 2
        X, Y = input
        train_input = X[:,:,1:num_train] , Y[:,:,1:num_train]
        test_input = X[:,:,num_train+1:num_train+num_test], Y[:,:,num_train+1:num_train+num_test]
        validation_input = X[:,:,num_train+num_test+1:end], Y[:,:,num_train+num_test+1:end]
    else
        X = input[1]
        train_input = (X[:,:,1:num_train],)
        test_input = (X[:,:,num_train+1:num_train+num_test],)
        validation_input = (X[:,:,num_train+num_test+1:end],)
    end

    # training
    optimal_params = deepcopy(Flux.params(model))

    optim = Flux.setup(Flux.Adam(learning_rate), model)
    loader = Flux.DataLoader(train_input, batchsize=batch_size, shuffle=true);

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
        for inp in loader
            loss, grads = Flux.withgradient(model) do m
                sum(m(inp...))
            end

            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
        end
        
        push!(losses, accumulated_loss / size(train_input[1])[end])
        push!(test_losses, mean(model(test_input...)))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
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

    output = mean(model(validation_input...))

    return output, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch, net=model)

end 


