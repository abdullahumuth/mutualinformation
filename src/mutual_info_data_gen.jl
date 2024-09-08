using Pkg 
Pkg.activate(".")

using BSON
using DataFrames
using Distributions
using HDF5


struct GeneratorTransformer{P <: Transformers.Layers.AbstractEmbedding}
    pos_embed::P
    a_embed::Chain
    decoder::Transformers.Layers.Transformer
    first_final_dense::Chain
    intermediate_dense::Chain
    b_embed::Chain
    encoder::Transformers.Layers.Transformer
    final_dense::Chain
    attention_mask
end

function GeneratorTransformer(;
    embedding_dim::Integer = 64, 
    head_num::Integer = 4, 
    head_dim::Integer = 8, 
    layer_num::Integer = 2,
    ffn_dim::Integer = 128,
    a_input_dim::Integer = 2,
    b_input_dim::Integer = 1,)


    decoder = Transformer(PreNormTransformerDecoderBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)
    encoder = Transformer(PreNormTransformerBlock, layer_num, head_num, embedding_dim, head_dim, ffn_dim)

    return GeneratorTransformer(
        SinCosPositionEmbed(embedding_dim) |> gpu,
        Chain(Dense(a_input_dim => ffn_dim), Dense(ffn_dim => embedding_dim)) |> gpu,
        decoder |> gpu,
        Chain(Dense(embedding_dim => ffn_dim), Dense(ffn_dim => a_input_dim)),
        Chain(Dense(a_input_dim => ffn_dim), Dense(ffn_dim => embedding_dim)) |> gpu,
        Chain(Dense(b_input_dim => ffn_dim), Dense(ffn_dim => embedding_dim)) |> gpu,
        encoder |> gpu,
        Chain(Dense(embedding_dim => ffn_dim), Dense(ffn_dim => a_input_dim)) |> gpu,
        NeuralAttentionlib.CausalMask())
end


function embedding(m::GeneratorTransformer, x; encoder=false)
    if encoder
        we = m.b_embed(x)
    else
        we = m.a_embed(x)
    end
    pe = m.pos_embed(we)
    return we .+ pe
end

function encoder_forward(m::GeneratorTransformer, y)
    e = embedding(m, y, encoder = true)
    t = m.encoder(e, nothing)
    return t.hidden_state
end

function cross_attend(m::GeneratorTransformer, input, encoder_output)
    e = embedding(m, input)
    t = m.decoder(e, encoder_output, m.attention_mask, nothing)
    return t.hidden_state
end

function generate(m, input_dim, seq_len, num_samples, y)
    x  = zeros((input_dim, seq_len, num_samples)) |> gpu
    x_given_y = zeros((input_dim, seq_len, num_samples)) |> gpu
    for i in 1:seq_len
        h = decoder_forward(m, x)
        h = m.first_final_dense(h)
        h = h[:,i,:]
        h = softmax(h)
        h = cpu(h)
        h = mapslices(x->rand(Categorical(x)), h, dims=1)
        h = gpu(h)
        h = reshape(Flux.onehotbatch(h, 1:input_dim), (input_dim, num_samples))
        x[:,i,:] = h
    end
    for i in 1:seq_len
        h = decoder_forward(m, x_given_y)
        h = m.first_final_dense(h)
        h = m.intermediate_dense(h)
        encoded = encoder_forward(m, y)
        h = cross_attend(m, h, encoded)
        h = m.final_dense(h)
        h = h[:,i,:]
        h = softmax(h)
        h = cpu(h)
        h = mapslices(x->rand(Categorical(x)), h, dims=1)
        h = gpu(h)
        h = reshape(Flux.onehotbatch(h, 1:input_dim), (input_dim, num_samples))
        x_given_y[:,i,:] = h
    end
    return x, x_given_y
end

function helper(x)
    x = log2(exp(1)).* sum(x, dims=2)
    x = reshape(x, (1,size(x)[3]))
    return x
end


function (m::GeneratorTransformer)(x, y)
    padded_x = pad_constant(x, (0,0,1,0,0,0), padding_constant)
    h = decoder_forward(m, padded_x)
    h = m.first_final_dense(h)
    hf = m.intermediate_dense(h)

    h = h[:,1:end-1,:]
    h = Flux.logitcrossentropy(h, x, agg = helper)

    encoded = encoder_forward(m,y)
    hf = cross_attend(m, hf, encoded)
    hf = m.final_dense(hf)
    hf = hf[:,1:end-1,:]
    hf = Flux.logitcrossentropy(hf, x, agg = helper)
    return h, hf
end



function evaluate(model, x, y)
    return mean.(model(x, y))
end

function generator(name, model_output_dim = 2, model_output_seq_len = 20, initially_generated_dim = 1, initially_generated_seq_len=2, num_samples=10000, seed=313)
    rng = Xoshiro(seed)
    m = GeneratorTransformer(a_input_dim = model_output_dim, b_input_dim = initially_generated_dim)
    y = (2 .* rand(rng, (initially_generated_dim, initially_generated_seq_len, num_samples)) .- 1) |> gpu
    x, x_given_y = generate(m, model_output_dim, model_output_seq_len, num_samples, y) |> gpu
    output = (x_given_y, y)


    mkdir_safe("data/inputs/$(name)")
    mkdir_safe("data/inputs/$(name)/models")
    mkdir_safe("data/inputs/$(name)/data")
    let 
        model = cpu(m)
        bson("data/inputs/$(name)/models/totally_generated_" * name * ".bson", Dict(:model => model))
    end
    h5open("data/inputs/$(name)/data/totally_generated_" * name * ".h5", "w") do file
        g = create_group(file, "data")
        g["x"] = cpu(output[1])
        g["y"] = cpu(output[2])
    end
    return (model = m, data = output)
end


function load_generated_data(name, num_samples)
    file = h5open("data/inputs/$(name)/data/totally_generated_" * name * ".h5", "r")
    data = read(file, "data")
    close(file)
    m = BSON.load("data/inputs/$(name)/models/totally_generated_" * name * ".bson")
    indices = randperm(Xoshiro(303), size(data["x"])[3])[1:num_samples]
    full_data = (data["x"][:,:,indices], data["y"][:,:,indices])

    return (model = m[:model], data = full_data)
end





function generation_experiment(name, model_output_dim = 2, model_output_seq_len = 20, initially_generated_dim = 1, initially_generated_seq_len=2, num_samples = 10000; seed=313, get_data_from = "", kwargs...)
    if get_data_from == ""
        get_data_from = name
        m = generator(name, model_output_dim, model_output_seq_len, initially_generated_dim, initially_generated_seq_len, num_samples, seed)
    else
        m = load_generated_data(get_data_from, num_samples)
    end
    mkdir_safe("data/inputs/$(get_data_from)")
    mkdir_safe("data/inputs/$(get_data_from)/results")

    gpu_model, gpu_data = m.model |> gpu, m.data |> gpu

    a = evaluate(gpu_model, gpu_data...; kwargs...)
    model_name = "totally_generated"

    result = DataFrame(Symbol(model_name) => a)
    CSV.write("data/inputs/$(get_data_from)/results/totally_generated_" * name * ".csv", result)
    println("entropy and the conditional entropy", " of the data called ", get_data_from, " is ", a)
end

#LET'S try

generation_experiment("total_generation_first", 2, 20, 1, 2, 2^16)