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
    b_input_dim::Integer = 0,
    gaussian_num::Integer = 0,)

    if gaussian_num == 0
        deembedding = Dense(embedding_dim => a_input_dim)
    else
        if a_input_dim != 1
            throw(ArgumentError("a_input_dim must be 1 when using gaussian mixtures"))
        end
        println("gaussian_num: ", gaussian_num)
        deembedding = Dense(embedding_dim => 3*gaussian_num)
    end

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
        deembedding |> gpu,
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



function log_weighted_gaussian(x, weight, μ, σ)
    σ_sq = 1e-6+softplus(σ) #σ^2
    return weight + ( -0.5 * (x - μ)^2 / σ_sq ) - 0.5*log(2*pi*σ_sq)
end



function log_gaussian_mix(ps, x)

    log_weights = logsoftmax(ps[:,1,:,:])
    gvals = log_weighted_gaussian.(x, log_weights, ps[:,2,:,:], ps[:,3,:,:])
    
    return logsumexp(gvals, dims=1)

end
# we pad to be able to calculate the unconditional probability
# with the final dense layer we calculate the probability of the next token == 1 (with 1d sigmoid input)
# after the final dense layer, we remove the prediction bit
# then we calculate the cross entropy (I just want to calculate p(x), that's why I choose h when x = 1 and 1-h when x = 0)
# we will need the sum of the logarithms of the probabilities, that would be the -log likelihood.
function (m::GeneralTransformer)(x, discrete = true)
    padded_x = pad_constant(x, (0,0,1,0,0,0), padding_constant) 
    h = decoder_forward(m, padded_x)
    h = m.final_dense(h)
    h = h[:,1:end-1,:]
    if discrete
        h = Flux.logitcrossentropy(h, x, agg = helper)
    else 
        # dimensions are (gaussian_num, 3, seq_len, batch_size)
        gm_params = reshape(h, (:, 3, size(x)[end-1:end]...))
        log_probs = log_gaussian_mix(gm_params, x)
        h = reshape(sum(log_probs, dims=2), (:))
    end
    return h
end


#conditional probability
function (m::GeneralTransformer)(x, y, discrete = true)
    
    padded_x = pad_constant(x, (0,0,1,0,0,0), padding_constant)
    encoded = encoder_forward(m,y)
    h = cross_attend(m, padded_x, encoded)
    h = m.final_dense(h)
    h = h[:,1:end-1,:]
    if discrete
        h = Flux.logitcrossentropy(h, x, agg = helper)
    else 
        gm_params = reshape(h, (:, 3, size(x)[end-1:end]...))
        log_probs = log_gaussian_mix(gm_params, x)
        h = reshape(sum(log_probs, dims=2), (:))
    end
    return h
end

Flux.@functor GeneralTransformer

function train(model, input...;
    learning_rate = 1e-4,
    max_epochs = 10000,
    batch_size = 128,
    test_fraction = 0.1,
    validation_fraction = 0.1,
    seed = 0,
    progress_bar = false,
    auto_stop = true)

    discrete = true
    (size(model.a_embed.:weight)[2] == 1) && (size(model.final_dense.:weight)[1] != 1) && (discrete = false)
    println("inputdim: ", size(model.a_embed.:weight)[2])
    println("outputdim: ", size(model.final_dense.:weight)[1])
    println("Discrete: ", discrete)

    Random.seed!(seed)

    # organize data

    num_samples = size(input[1])[3]
    num_test = Int(floor(test_fraction * num_samples))
    num_valid = Int(floor(validation_fraction * num_samples))
    num_train = num_samples - num_test - num_valid

    
    if length(input) == 2
        X, Y = input
        println("X: ", size(X))
        println("Y: ", size(Y))
        train_input = X[:,:,1:num_train] , Y[:,:,1:num_train]
        test_input = X[:,:,num_train+1:num_train+num_test], Y[:,:,num_train+1:num_train+num_test]
        validation_input = X[:,:,num_train+num_test+1:end], Y[:,:,num_train+num_test+1:end]
    else
        X = input[1]
        println("X: ", size(X))
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
    times = [Float64(0)]
    min_epoch = 0

    epochs = 1:max_epochs
    progress = nothing
    if progress_bar
        progress = Progress(max_epochs; dt=1.0)
    end
    early_stopper = Flux.early_stopping(()->test_losses[end], 500; init_score = Inf)
    for epoch in epochs
        accumulated_loss = 0
        for inp in loader
            (epoch + 1) % 100 == 0 && (t1 = time()) 
            loss, grads = Flux.withgradient(model) do m
                sum(m(inp..., discrete = discrete))
            end
            accumulated_loss += loss
            Flux.update!(optim, model, grads[1])
            if (epoch + 1) % 100 == 0
                t2 = time()
                push!(times, t2-t1)
            end
        end
        
        push!(losses, accumulated_loss / size(train_input[1])[end])
        push!(test_losses, mean(model(test_input..., discrete = discrete)))

        if size(test_losses)[1]>1
            if min(test_losses[1:end-1]...) > test_losses[end]
                optimal_params = deepcopy(Flux.params(model))
                min_epoch = epoch
            end

            auto_stop && early_stopper() && break
        end

        if progress_bar
            next!(progress)
        end
    end

    Flux.loadparams!(model, optimal_params)

    output = mean(model(validation_input..., discrete = discrete))

    length(times) > 1 && popfirst!(times)

    return output, (train_losses = Float32.(losses), test_losses = Float32.(test_losses), min_epoch = min_epoch, net=model, avg_time = mean(times))

end 


