using Random
using Statistics
using Flux
using Flux.Functors
using Transformers
using NeuralAttentionlib
using ProgressMeter

struct GeneralTransformer{P <: Transformers.Layers.AbstractEmbedding, 
                                  T <: Transformers.Layers.Transformer}
    general_embed::Dense
    pos_embed::P
    trf::T
    attention_mask
end

function GeneralTransformer(
    hidden_dim::Integer = 64, 
    head_num::Integer = 4, 
    head_dim::Integer = 8, 
    layer_num::Integer = 2,
    ffn_dim::Integer = 128,
    is_encoder::Bool = false,
    has_cross_attention::Bool = false)
    
    return GeneralTransformer(
        Dense(1 => hidden_dim) |> gpu,
        SinCosPositionEmbed(hidden_dim) |> todevice,

        if has_cross_attention 
        Transformer(PreTransformerBlock, layer_num, head_num, hidden_dim, head_dim, ffn_dim) |> todevice 
        else 
        Transformer(PreTransformerDecoderBlock, layer_num, head_num, hidden_dim, head_dim, ffn_dim) |> todevice end,

        if is_encoder nothing else NeuralAttentionlib.CausalMask() end,
    )

end


function embedding(m::GeneralTransformer, x)
    we = m.general_embed(x)
    pe = m.pos_embed(we)
    return we .+ pe
end

function encoder_forward(m::GeneralTransformer, input)
    e = embedding(m, input)
    t = m.trf(e, m.attention_mask)
    return t
end