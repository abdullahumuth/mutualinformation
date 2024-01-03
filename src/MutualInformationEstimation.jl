module MutualInformationEstimation

include("./entropy.jl")
using .Entropy
export compute_entropy,
        compute_conditional_entropy

end # module MutualInformationEstimation
