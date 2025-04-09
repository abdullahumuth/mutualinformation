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
function calculate_p_b_given_a(model::GeneralTransformer, b, a)
    # Convert b to one-hot encoding if it's not already in the right format
    if ndims(b) == 2
        input_dim = size(model.final_dense.weight)[1]
        seq_len = size(b, 1)
        
        # Create one-hot encoding in a single operation
        b_onehot = Flux.onehotbatch(reshape(b, :), 1:input_dim)
        b_onehot = reshape(b_onehot, (input_dim, seq_len, :)) |> gpu
    else
        b_onehot = b |> gpu
    end
    
    # Process a without reshaping (assuming a is already properly formatted)
    a_tensor = a |> gpu
    
    # Get conditional probabilities directly
    probs = model(b_onehot, a_tensor, find_probs=true, discrete=true)
    
    return cpu(probs)
end

"""
Calculate p(b) by integrating p(a) * p(b|a) over all values of a
"""
function calculate_p_b(model::GeneralTransformer, b, p_a::MixtureGaussianPDF;
                      a_range=(-10.0, 10.0), n_samples=100)
    # Create a grid of a values
    a_values = LinRange(a_range[1], a_range[2], n_samples)
    
    # Calculate p(b|a) for each a value (more efficiently with broadcasting)
    a_tensor = reshape(collect(a_values), 1, 1, :)
    repeated_b = repeat(b, outer=(1, 1, n_samples))
    p_b_given_a_values = calculate_p_b_given_a(model, repeated_b, a_tensor)
    
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
    
    return p_b
end

"""
Sample from the joint distribution p(a,b)
"""
function sample_joint_distribution(model::GeneralTransformer, p_a::MixtureGaussianPDF,
                                 b_dims::Int, seq_len::Int, n_a_samples::Int, samples_per_a::Int)
    # Total number of samples
    n_total_samples = n_a_samples * samples_per_a
    
    # Sample from p(a)
    a_samples_unique = sample_from_p_a(p_a, n_a_samples)
    # Repeat each a value samples_per_a times
    a_samples = repeat(a_samples_unique, inner=samples_per_a)

    b_samples = CUDA.zeros(Int, b_dims, seq_len, n_total_samples)
    
    # Get input dimension from the model
    input_dim = size(model.final_dense.weight)[1]
    
    # Process in batches for GPU memory efficiency
    batch_size = 64
    
    for batch_start in 1:batch_size:n_total_samples
        # Get the current batch range
        batch_end = min(batch_start + batch_size - 1, n_total_samples)
        current_batch_size = batch_end - batch_start + 1
        batch_idx = batch_start:batch_end
        
        # Prepare a for the model
        a_batch = reshape(a_samples[batch_idx], 1, 1, :) |> gpu
        
        # Generate samples and copy directly on GPU
        samples = generate_samples(model, input_dim, seq_len, current_batch_size, a_batch)
        b_samples[:,:, batch_idx] = samples
    end
    
    # Return both the unique a samples and their repetitions, along with b samples
    return a_samples_unique, a_samples, cpu(b_samples)
end


"""
Calculate entropy of b: E(b) = -∑ p(b) log p(b)
Either exact calculation for small spaces or Monte Carlo for large ones.
"""
function calculate_entropy_b(model::GeneralTransformer, p_a::MixtureGaussianPDF;
                           n_samples=1000, seq_len=10, input_dim=nothing,
                           show_progress=true)
    
    # Input validation and setup
    input_dim = input_dim === nothing ? size(model.final_dense.weight, 1) : input_dim
    total_sequences = input_dim^seq_len
    
    @info "Sequence space size: $total_sequences"
    
    # Choose method based on space size
    if total_sequences > 10000
        @info "Using Monte Carlo estimation"
        return monte_carlo_entropy(model, p_a, n_samples, seq_len, input_dim; show_progress)
    else
        @info "Using exact calculation"
        return exact_entropy(model, p_a, seq_len, input_dim; show_progress)
    end
end

"""
Exact entropy calculation for small sequence spaces
"""
function exact_entropy(model, p_a, seq_len, input_dim; show_progress=true)
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
        
        p_b = calculate_p_b(model, sequence_onehot, p_a)
        
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
function monte_carlo_entropy(model, p_a, n_samples, seq_len, input_dim; 
                           show_progress=true, batch_size=100)
    
    # Initialize storage
    n_samples_per_a = floor(Int, sqrt(batch_size))
    b_samples_per_a = floor(Int, n_samples / n_samples_per_a)
    batch_size = n_samples_per_a * b_samples_per_a
    entropy = 0.0
    seen = Dict{String, Float64}()
    log_probs = Float64[]  # Store for confidence calculation
    
    # Setup batching
    n_batches = ceil(Int, n_samples / batch_size)
    prog = show_progress ? Progress(n_batches, "Monte Carlo estimation: ") : nothing
    for batch in 1:n_batches
        # Sample from joint distribution
        _, _, b_samples = sample_joint_distribution(
            model, p_a, input_dim, seq_len, 
            1, batch_size
        )

        # shuffle the samples to avoid bias
        b_samples = b_samples[:, :, randperm(batch_size)]
        
        # Process each sample in batch
        for i in 1:batch_size
            sequence = b_samples[:,:,i]
            key = string(sequence)
            
            # Get or calculate p(b)
            p_b = get!(seen, key) do
                calculate_p_b(model, sequence, p_a)
            end
            
            # Store log probability for confidence calculation
            if p_b > 0
                push!(log_probs, -log2(p_b))
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
"""
Main function to run the complete analysis
"""
function analyze_conditional_probability(
    model_path::String = nothing;
    p_a_params = Dict(
        "weights" => [0.5, 0.5],
        "means" => [0.0, 3.0],
        "stds" => [1.0, 0.5]
    ),
    seq_len = 10,
    input_dim = 2,
    n_samples_entropy = 1000,
    a_range = (-10.0, 10.0)
)
    # Load model if path provided
    if model_path !== nothing
        model = Flux.loadmodel(model_path)
    else
        raise(ArgumentError("Model path is required"))
    end
    
    # Define probability distribution p(a)
    p_a = MixtureGaussianPDF(
        p_a_params["weights"],
        p_a_params["means"],
        p_a_params["stds"]
    )
    
    # Sample from p(a)
    println("Sampling from p(a)...")
    a_samples = sample_from_p_a(p_a, 10)
    println("Sample a values: ", a_samples)
    
    # Generate a test binary sequence
    test_sequence = rand(1:input_dim, seq_len)
    println("Test binary sequence: ", test_sequence)
    
    # Calculate p(b|a) for the test sequence
    println("\nCalculating p(b|a) for test sequence...")
    p_b_given_a = calculate_p_b_given_a(model, reshape(test_sequence, (seq_len, 1)), [a_samples[1]])
    println("p(b|a=", a_samples[1], ") = ", p_b_given_a[1])
    
    # Calculate p(b) by integration
    println("\nCalculating p(b) by integration...")
    p_b = calculate_p_b(model, reshape(test_sequence, (seq_len, 1)), p_a, a_range=a_range)
    println("p(b) = ", p_b)
    
    # Sample from joint distribution
    println("\nSampling from joint distribution p(a,b)...")
    joint_a_samples, joint_b_samples = sample_joint_distribution(model, p_a, seq_len, 5)
    for i in 1:5
        println("Sample ", i, ": a = ", joint_a_samples[i], ", b = ", joint_b_samples[:, i])
    end
    
    # Calculate entropy
    println("\nCalculating entropy of b...")
    entropy = calculate_entropy_b(model, p_a, n_samples=n_samples_entropy, seq_len=seq_len, input_dim=input_dim)
    println("Entropy of b: ", entropy, " bits")
    
    return Dict(
        "model" => model,
        "p_a" => p_a,
        "test_p_b_given_a" => p_b_given_a,
        "test_p_b" => p_b,
        "joint_samples" => (joint_a_samples, joint_b_samples),
        "entropy" => entropy
    )
end

# Example usage
function example()
    # Define parameters for p(a) as a mixture of Gaussians
    p_a_params = Dict(
        "weights" => [0.7, 0.3],
        "means" => [-2.0, 2.0],
        "stds" => [1.0, 0.5]
    )
    
    # Run the analysis
    results = analyze_conditional_probability(
        "data\\inputs\\production_generation\\models\\generated_production_generation.bson",  # No model path, create a new model
        p_a_params = p_a_params,
        seq_len = 8,
        input_dim = 2,
        n_samples_entropy = 500,
        a_range = (-10.0, 10.0)
    )
    
    return results
end

# To run the example:
#results = example()