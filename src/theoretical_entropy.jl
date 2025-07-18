using Random
using Statistics
using Flux
using Distributions
using QuadGK
using CUDA
using ProgressMeter

include("./transformer copy.jl")

# Assuming all the code you provided is already included in your Julia environment
# The following code extends that functionality

"""
Define probability distribution for p(a)
"""
struct MixtureGaussianPDF
    weights::Vector{Float64}
    means::Vector{Float64}
    stds::Vector{Float64}
    
    function MixtureGaussianPDF(weights::Vector{Float64}, means::Vector{Float64}, stds::Vector{Float64})
        if length(weights) != length(means) || length(weights) != length(stds)
            throw(ArgumentError("Weights, means, and stds must have the same length"))
        end
        if !isapprox(sum(weights), 1.0, atol=1e-6)
            throw(ArgumentError("Weights must sum to 1.0"))
        end
        return new(weights, means, stds)
    end
end

function (pdf::MixtureGaussianPDF)(a::Float64)
    result = sum(pdf.weights .* pdf_gaussian.(a, pdf.means, pdf.stds))
    return result
end

function pdf_gaussian(x, μ, σ)
    σ = max(σ, 1e-4)  # Avoid division by zero
    return exp(-0.5 * (x - μ)^2 / (σ^2)) / (σ * sqrt(2π))
end


using Plots

"""
Plot the mixture Gaussian PDF and its components.
range_min and range_max define the x-axis range.
n_points is the number of points to plot.
"""
function plot_mixture_gaussian(pdf::MixtureGaussianPDF; 
                             range_min=-2, range_max=5, 
                             n_points=500)
    
    x = range(range_min, range_max, length=n_points)
    
    # Plot individual components
    p = plot(xlabel="x", ylabel="Probability Density")
    for i in 1:length(pdf.weights)
        y = pdf.weights[i] .* pdf_gaussian.(x, pdf.means[i], pdf.stds[i])
        plot!(p, x, y, 
              label="Component $i (w=$(pdf.weights[i]), μ=$(pdf.means[i]), σ=$(pdf.stds[i]))",
              linestyle=:dash)
    end
    
    # Plot combined PDF
    y_total = pdf.(x)
    plot!(p, x, y_total, 
          label="Combined PDF", 
          linewidth=2)
    
    return p
end

# Usage:
# plot_mixture_gaussian(a)

# Basic plot with default range
# p = plot_mixture_gaussian(a)

# Custom range
# p = plot_mixture_gaussian(a, range_min=-1, range_max=6)

# Save the plot if needed
# savefig(p, "mixture_gaussian.png")



"""
Sample from the mixture of Gaussians
"""
function sample_from_p_a(pdf::MixtureGaussianPDF, n_samples::Int)
    mixture = MixtureModel(Normal.(pdf.means, pdf.stds), pdf.weights)
    return rand(mixture, n_samples)
end

"""
Calculate p(b|a) using the trained transformer model
"""
function calculate_p_b_given_a(model::GeneralTransformer, b, a; return_log_prob = false)
    # Convert b to one-hot encoding if it's not already in the right format
    if ndims(b) == 2
        input_dim = size(model.final_dense.weight)[1]
        seq_len = size(b, 1)
        
        # Create one-hot encoding in a single operation
        b_onehot = Flux.onehotbatch(reshape(b, :), 1:input_dim)
        b_onehot = reshape(b_onehot, (input_dim, seq_len, :)) |> gpu
    elseif ndims(b) != 3
        throw(ArgumentError("b must be a 2D or 3D array"))
    else
        b_onehot = b |> gpu
    end
    
    # Process a without reshaping (assuming a is already properly formatted)
    a_tensor = a |> gpu
    
    # Get conditional probabilities directly
    probs = model(b_onehot, a_tensor, find_probs=true, discrete=true, return_log_prob=return_log_prob)
    return cpu(probs)
end

"""
Calculate p(b) by integrating p(a) * p(b|a) over all values of a
"""
function calculate_p_b(model::GeneralTransformer, b, p_a::MixtureGaussianPDF; return_log_prob = false,
                      a_range=(-10.0, 10.0), n_samples=100)
    # Create a grid of a values
    a_values = LinRange(a_range[1], a_range[2], n_samples)
    
    # Calculate p(b|a) for each a value (more efficiently with broadcasting)
    a_tensor = reshape(collect(a_values), 1, 1, :)
    repeated_b = repeat(b, outer=(1, 1, n_samples))
    p_b_given_a_values = calculate_p_b_given_a(model, repeated_b, a_tensor; return_log_prob=false)
    # Define the integrand function as a vector operation
    integrand(a) = p_a(a) * linear_interpolation(a, a_values, p_b_given_a_values)
    
    # Interpolation function to get p(b|a) for arbitrary a
    function linear_interpolation(a, grid, values)
        if a <= grid[1]
            return values[1]
        elseif a >= grid[end]
            return values[end]
        else
            idx = searchsortedfirst(grid, a)
            if idx == 1
                idx = 2
            end
            t = (a - grid[idx-1]) / (grid[idx] - grid[idx-1])
            return values[idx-1] * (1-t) + values[idx] * t
        end
    end
    
    # Perform numerical integration
    p_b, _ = quadgk(integrand, a_range[1], a_range[2])
    
    if return_log_prob
        p_b = log2(p_b)
    end

    return p_b
end


function calculate_p_b_by_batches(model::GeneralTransformer, b, p_a::MixtureGaussianPDF; return_log_prob = false,
    a_range=(-10.0, 10.0), n_samples=100)
    p_b = zeros(Float64, 1, 1, size(b, 3))
    for i in axes(b, 3)
        b_i = b[:, :, i]
        p_b_i = calculate_p_b(model, b_i, p_a; return_log_prob=return_log_prob, a_range=a_range, n_samples=n_samples)
        p_b[:,:,i] .= p_b_i
    end
    return p_b
end

"""
Sample from the joint distribution p(a,b)
"""
function sample_joint_distribution(model::GeneralTransformer, p_a::MixtureGaussianPDF,
                                 b_dims::Int, seq_len::Int, n_samples::Int)
    # Sample from p(a)
    a_samples = sample_from_p_a(p_a, n_samples)
    b_samples = CUDA.zeros(Int, b_dims, seq_len, n_samples)
    
    # Get input dimension from the model
    input_dim = size(model.final_dense.weight)[1]
    
    # Process in batches for GPU memory efficiency
    batch_size = 64
    
    for batch_start in 1:batch_size:n_samples
        # Get the current batch range
        batch_end = min(batch_start + batch_size - 1, n_samples)
        current_batch_size = batch_end - batch_start + 1
        batch_idx = batch_start:batch_end
        
        # Prepare a for the model
        a_batch = reshape(a_samples[batch_idx], 1, 1, :) |> gpu
        
        # Generate samples and copy directly on GPU
        samples = generate_samples(model, input_dim, seq_len, current_batch_size, a_batch)
        b_samples[:,:, batch_idx] = samples
    end
    
    # Return both the unique a samples and their repetitions, along with b samples
    return a_samples, cpu(b_samples)
end


"""
Calculate entropy of b: E(b) = -∑ p(b) log p(b)
Either exact calculation for small spaces or Monte Carlo for large ones.
"""
function calculate_entropy_b(model::GeneralTransformer, p_a::MixtureGaussianPDF;
                           n_samples=1000, seq_len=10, input_dim=nothing, a_range=(-10, 10), a_integration_samples=200,
                           show_progress=true)
    
    # Input validation and setup
    input_dim = input_dim === nothing ? size(model.final_dense.weight, 1) : input_dim
    total_sequences = input_dim^seq_len
    
    @info "Sequence space size: $total_sequences"
    
    # Choose method based on space size
    if total_sequences > 10000
        @info "Using Monte Carlo estimation"
        return monte_carlo_entropy(model, p_a, n_samples, seq_len, input_dim, a_range, a_integration_samples; show_progress)
    else
        @info "Using exact calculation"
        return exact_entropy(model, p_a, seq_len, input_dim, a_range, a_integration_samples; show_progress)
    end
end

"""
Exact entropy calculation for small sequence spaces
"""
function exact_entropy(model, p_a, seq_len, input_dim, a_range, a_integration_samples; show_progress=true)
    total_sequences = input_dim^seq_len
    entropy = 0.0
    
    # Progress meter
    prog = show_progress ? Progress(total_sequences, "Computing exact entropy: ") : nothing
    
    # Exact calculation
    for idx in 0:(total_sequences-1)
        # Convert number to sequence and one-hot encode
        sequence = digits(idx, base=input_dim, pad=seq_len) .+ 1
        sequence_onehot = Int.(reshape(Flux.onehotbatch(sequence, 1:input_dim), 
                                     (input_dim, seq_len, 1))) |> gpu
        
        p_b = calculate_p_b(model, sequence_onehot, p_a; a_range = a_range, n_samples = a_integration_samples)
        
        # Update entropy
        if p_b > 0
            entropy -= p_b * log2(p_b)
        end
        
        # Update progress
        show_progress && next!(prog)
    end
    
    return entropy
end

"""
Monte Carlo approximation of entropy with confidence estimation
"""
function monte_carlo_entropy(model, p_a, n_samples, seq_len, input_dim, a_range, a_integration_samples; conditional=false,
                           show_progress=true, batch_size=128)
    

    # Initialize storage
    entropy = 0.0
    seen = Dict{String, Float64}()
    log_probs = Float64[]  # Store for confidence calculation
    
    # Setup batching
    n_batches = ceil(Int, n_samples / batch_size)
    prog = show_progress ? Progress(n_batches, "Monte Carlo estimation: ") : nothing
    for batch in 1:n_batches
        # Sample from joint distribution                                                                      
        a_samples, b_samples = sample_joint_distribution(
            model, p_a, input_dim, seq_len, 
            batch_size
        )

        # shuffle the samples to avoid bias
        a_samples = a_samples[randperm(batch_size)]
        b_samples = b_samples[:, :, randperm(batch_size)]
        if conditional
            # Calculate conditional entropy
            log_prob = calculate_p_b_given_a(model, b_samples, a_samples; return_log_prob=true)
        else
            log_prob = calculate_p_b_by_batches(model, b_samples, p_a; a_range = a_range, n_samples = a_integration_samples, return_log_prob=true)
        end
        
        # Store log probability for confidence calculation
        # Note: log_prob is already negative log probability for entropy
        for one_log_prob in log_prob
            if one_log_prob > -Inf
                push!(log_probs, -one_log_prob)  # Convert to -log p(b)
            else
                println("Warning: Encountered -Inf log probability, skipping this sample")
            end
        end
        show_progress && next!(prog)
    end
    
    # Calculate entropy and confidence interval
    mean_entropy = mean(log_probs)
    std_error = std(log_probs) / sqrt(length(log_probs))
    confidence_95 = 1.96 * std_error
    
    @info "Entropy estimation complete" mean_entropy confidence_interval="±$confidence_95"
    
    return mean_entropy, confidence_95
end

