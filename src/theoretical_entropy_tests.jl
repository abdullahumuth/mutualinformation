include("./theoretical_entropy.jl")
using Plots
using BSON
using Distributions
using StatsBase

# Helper function to display and save plots
function display_and_save(plot_obj, filename="plot.png")
    try
        display(plot_obj)
    catch e
        println("Display failed: $e")
    end
    
    try
        savefig(plot_obj, filename)
        println("Plot saved as '$filename'")
    catch e
        println("Save failed: $e")
    end
end

# Test different plotting backends
function test_plotting_backends()
    println("Testing plotting backends...")
    
    # Test simple plot
    x = 1:10
    y = x.^2
    
    # Try GR backend (default)
    try
        gr()
        p = plot(x, y, title="Test Plot - GR Backend")
        display_and_save(p, "test_gr.png")
        println("✓ GR backend works")
    catch e
        println("✗ GR backend failed: $e")
    end
    
    # Try PlotlyJS backend
    try
        plotlyjs()
        p = plot(x, y, title="Test Plot - PlotlyJS Backend")
        display_and_save(p, "test_plotlyjs.html")
        println("✓ PlotlyJS backend works")
    catch e
        println("✗ PlotlyJS backend failed: $e")
    end
    
    # Reset to GR
    gr()
end

function GaussianPDFTest()
    a = MixtureGaussianPDF([0.5, 0.5],[0.0, 3.0],[1.0, 0.5])
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
    p = plot_mixture_gaussian(a)
    display(p)
end

function sampleFromATest()
    function plot_samples_and_pdf(pdf::MixtureGaussianPDF, n_samples::Int=1000)

        samples = sample_from_p_a(pdf, n_samples)

        x = range(minimum(samples)-1, maximum(samples)+1, length=200)



        plot(x, pdf.(x), label="PDF", linewidth=2)

        histogram!(samples, normalize=true, alpha=0.3, label="Samples")

    end
    a = MixtureGaussianPDF([0.5, 0.5],[0.0, 3.0],[1.0, 0.5])
    p = plot_samples_and_pdf(a, 1000)
    display(p)
end

function test_p_b_given_a()

    model_path = "data/inputs/production_generation/models/generated_production_generation.bson"
    # Load the model

    model = BSON.load(model_path)[:model] |> gpu
    p_a = MixtureGaussianPDF([0.5, 0.5],[0.0, 3.0],[1.0, 0.5])
    # b has (input_dim, seq_len, num_samples) : (2, 20, whatever we want)
    # one hot encoded of course. lets start with 5 samples.
    # test 5 random samples (one hot encoded) of b.
    b = rand(Categorical([0.5, 0.5]), 1, 20, 5)
    b = Int.(reshape(Flux.onehotbatch(b, 1:2), (2, 20, 5))) |> gpu
    a = sample_from_p_a(p_a, 5) 
    a = reshape(a, (1, 1, 5)) |> gpu

    # Calculate conditional probabilities

    probs = calculate_p_b_given_a(model, b, a)
    println("Input shapes:")
    println("a: $(size(a))")
    println("b: $(size(b))")
    println("\nOutput probabilities shape: $(size(probs))")
    println("probabilities: $(probs)")
    return probs
end

function test_calculate_p_b()
    # Load the model
    model_path = "data/inputs/production_generation/models/generated_production_generation.bson"
    model = BSON.load(model_path)[:model] |> gpu
    
    # Create a mixture Gaussian PDF
    p_a = MixtureGaussianPDF([0.5, 0.5], [-0.2, 0.2], [0.1, 0.1])
    
    # Create a test sequence b (one-hot encoded)
    # Testing with a single sequence
    b = Int.(reshape(Flux.onehotbatch(rand(Categorical([0.05, 0.95]), 1, 20), 1:2), (2, 20, 1))) |> gpu
    
    a_range = (-1.0, 1.0)
    n_points = 100


    # Calculate p(b)
    p_b = calculate_p_b(model, b, p_a; a_range=a_range, n_samples=n_points)
    
    # Print results
    println("Test sequence b shape: $(size(b))")
    println("Calculated p(b): $(p_b)")
    
    # Optional: Create a visualization to show the integration
    
    a_values = LinRange(a_range[1], a_range[2], n_points)
    
    # Calculate p(b|a) for visualization
    a_tensor = reshape(collect(a_values), 1, 1, :)
    b_repeated = repeat(b, outer=(1, 1, n_points))
    p_b_given_a_values = calculate_p_b_given_a(model, b_repeated, a_tensor)
    
    # Plot p(a) and p(b|a) and p(a,b)
    p = plot(layout=(3,1))
    
    # Plot p(a)
    plot!(p[1], a_values, p_a.(a_values), 
          label="p(a)", 
          title="p(a)", 
          xlabel="a", 
          ylabel="Probability Density")
    
    # Plot p(b|a)
    plot!(p[2], a_values, vec(p_b_given_a_values), 
          label="p(b|a)", 
          title="p(b|a)", 
          xlabel="a", 
          ylabel="Probability")
    # Plot p(a,b) = p(a) * p(b|a)
    p_ab = p_a.(a_values) .* vec(p_b_given_a_values)
    plot!(p[3], a_values, p_ab, 
          label="p(a,b)", 
          title="p(a,b)", 
          xlabel="a", 
          ylabel="Probability")
    
    # Add a legend
    plot!(p, legend=:topright)
    


    display(p)
    
    return p_b
end

function test_sample_joint_distribution()
    # Load the model
    model_path = "data/inputs/production_generation/models/generated_production_generation.bson"
    model = BSON.load(model_path)[:model] |> gpu
    
    # Create a mixture Gaussian PDF
    p_a = MixtureGaussianPDF([0.5, 0.5], [-0.2, 0.2], [0.1, 0.1])
    
    # Parameters for sampling
    b_dims = 2  # assuming binary sequences
    seq_len = 20
    n_samples = 10000 
    
    # Sample from joint distribution
    a_samples, b_samples = sample_joint_distribution(
        model, p_a, b_dims, seq_len, n_samples
    )
    
    # Print basic information
    println("Total number of samples: $(size(b_samples, 2))")
    println("Sequence length: $(size(b_samples, 1))")
    
    # Create visualizations
    p = plot(layout=(2,1), size=(800, 800))
    
    # Plot 1: Distribution of a values
    histogram!(p[1], a_samples, 
              bins=50, 
              normalize=true, 
              alpha=0.6,
              label="Sampled a distribution",
              title="Distribution of a values")
    
    # Add the true p(a) curve
    x_range = range(minimum(a_samples)-0.5, maximum(a_samples)+0.5, length=200)
    plot!(p[1], x_range, p_a.(x_range), 
          label="True p(a)", 
          linewidth=2)
    
    # Plot 2: Heatmap of b sequences
    # Take first 100 sequences for visualization

    b_samples_indices = b_samples[1,:,1:100] |> cpu
    random_indices = randperm(size(b_samples_indices, 2))[1:100]
    heatmap!(p[2], b_samples_indices[:, random_indices],
             title="First 100 generated sequences",
             xlabel="Sample index",
             ylabel="Sequence position",
             colorbar_title="Token value")
    
    # Optional: Add additional analysis
    # For example, calculate and plot token distributions at different positions
    
    # Display the plot
    display(p)
    
    # Return samples for further analysis if needed
    return a_samples, b_samples
end

function test_calculate_entropy_b()
    # Load the model
    model_path = "data/inputs/production_generation/models/generated_production_generation.bson"
    model = BSON.load(model_path)[:model] |> gpu
    
    # Create a mixture Gaussian PDF for testing
    p_a = MixtureGaussianPDF([0.4, 0.6], [-0.3, 0.3], [0.15, 0.1])
    
    # Test parameters
    input_dims = [2]  # Test with binary and ternary sequences
    seq_lengths = [5, 10, 20]  # Test with different sequence lengths

    a_range  = (-10, 10)
    n_a_points = 200
    
    # Store results for comparison
    results = Dict{Tuple{Int,Int,String},Any}()
    
    # Run tests for different combinations
    for input_dim in input_dims
        for seq_len in seq_lengths
            println("\n=== Testing with input_dim=$input_dim, seq_len=$seq_len ===")
            
            # Decide on method based on space size
            total_sequences = input_dim^seq_len
            if total_sequences <= 10000
                # Test exact calculation
                println("Running exact entropy calculation...")
                @time exact_result = exact_entropy(model, p_a, seq_len, input_dim, a_range, n_a_points)
                results[(input_dim, seq_len, "exact")] = exact_result
                println("Exact entropy: $exact_result bits")
                
                # Test Monte Carlo with different sample sizes
                for n_samples in [1000, 5000, 10000]
                    println("Running Monte Carlo with $n_samples samples...")
                    @time mc_result, confidence = monte_carlo_entropy(
                        model, p_a, n_samples, seq_len, input_dim, a_range, n_a_points; show_progress=false)
                    results[(input_dim, seq_len, "monte_carlo_$n_samples")] = (mc_result, confidence)
                    println("Monte Carlo entropy: $mc_result ± $confidence bits")
                    
                    # Calculate relative error
                    relative_error = abs(mc_result - exact_result) / exact_result * 100
                    println("Relative error: $(round(relative_error, digits=2))%")
                    
                    # Check if exact result is within confidence interval
                    if abs(mc_result - exact_result) <= confidence
                        println("✓ Exact result is within 95% confidence interval")
                    else
                        println("✗ Exact result is outside 95% confidence interval")
                    end
                end
            else
                # For large spaces, only run Monte Carlo
                println("Sequence space too large for exact calculation")
                
                # Test Monte Carlo with different sample sizes for convergence analysis
                previous_result = nothing
                for n_samples in [1000, 5000, 10000, 20000]
                    println("Running Monte Carlo with $n_samples samples...")
                    @time mc_result, confidence = monte_carlo_entropy(
                        model, p_a, n_samples, seq_len, input_dim, a_range, n_a_points; show_progress=false)
                    results[(input_dim, seq_len, "monte_carlo_$n_samples")] = (mc_result, confidence)
                    println("Monte Carlo entropy: $mc_result ± $confidence bits")
                    
                    # Check convergence if we have a previous result
                    if previous_result !== nothing
                        diff = abs(mc_result - previous_result)
                        println("Change from previous sample size: $diff bits")
                    end
                    previous_result = mc_result
                end
            end
        end
    end
    
    # Visualization of results
    println("\n=== Summary of Results ===")
    for input_dim in input_dims
        for seq_len in seq_lengths
            println("\nInput Dim: $input_dim, Sequence Length: $seq_len")
            
            # Get exact result if available
            exact_key = (input_dim, seq_len, "exact")
            has_exact = haskey(results, exact_key)
            exact_val = has_exact ? results[exact_key] : NaN
            
            # Print exact result
            if has_exact
                println("Exact entropy: $exact_val bits")
                
                # Create data for convergence plot
                mc_results = []
                mc_errors = []
                sample_sizes = []
                
                # Collect Monte Carlo results
                for n_samples in [1000, 5000, 10000]
                    mc_key = (input_dim, seq_len, "monte_carlo_$n_samples")
                    if haskey(results, mc_key)
                        mc_val, confidence = results[mc_key]
                        rel_error = abs(mc_val - exact_val) / exact_val * 100
                        push!(mc_results, mc_val)
                        push!(mc_errors, rel_error)
                        push!(sample_sizes, n_samples)
                        
                        println("Monte Carlo ($n_samples samples): $mc_val ± $confidence bits (Error: $(round(rel_error, digits=2))%)")
                    end
                end
                
                # Create convergence plot if we have plotting capability
                if @isdefined(plot)
                    p = plot(sample_sizes, mc_errors, 
                        title="Convergence Analysis (dim=$input_dim, len=$seq_len)",
                        xlabel="Number of Samples",
                        ylabel="Relative Error (%)",
                        marker=:circle,
                        label="Error vs Exact",
                        legend=:topright,
                        yaxis=:log
                    )
                    # Add horizontal line for 1% error
                    hline!(p, [1.0], linestyle=:dash, color=:red, label="1% Error")
                    display(p)
                end
            else
                println("No exact calculation available (sequence space too large)")
                
                # Collect Monte Carlo results
                for n_samples in [1000, 5000, 10000, 20000]
                    mc_key = (input_dim, seq_len, "monte_carlo_$n_samples")
                    if haskey(results, mc_key)
                        mc_val, confidence = results[mc_key]
                        println("Monte Carlo ($n_samples samples): $mc_val ± $confidence bits")
                    end
                end
            end
        end
    end
    
    # Test additional helpful utilities
    #test_joint_distribution_consistency(model, p_a)
    
    return results
end

"""
Test consistency between entropy and joint sampling
"""
function test_joint_distribution_consistency(model, p_a; 
                                            input_dim=2, 
                                            seq_len=10, 
                                            n_samples=10000)
    println("\n=== Testing Consistency Between Entropy and Joint Sampling ===")
    
    # Sample from joint distribution
    a_samples, b_samples = sample_joint_distribution(
        model, p_a, input_dim, seq_len, n_samples
    )
    
    # Compute empirical entropy from samples
    println("Computing empirical entropy from samples...")
    
    # Count occurrences of each sequence
    sequence_counts = Dict{String, Int}()
    total_samples = size(b_samples, 3)
    
    for i in 1:total_samples
        sequence = b_samples[:,:,i]
        key = string(sequence)
        sequence_counts[key] = get(sequence_counts, key, 0) + 1
    end
    
    # Calculate empirical entropy
    empirical_entropy = 0.0
    for (_, count) in sequence_counts
        p = count / total_samples
        empirical_entropy -= p * log2(p)
    end
    
    println("Empirical entropy from samples: $empirical_entropy bits")
    n_a_samples = 200
    # Calculate Monte Carlo entropy for comparison
    println("Calculating Monte Carlo entropy...")
    mc_entropy, confidence = monte_carlo_entropy(
        model, p_a, 5000, seq_len, input_dim, a_range, n_a_samples; show_progress=false)
    
    println("Monte Carlo entropy: $mc_entropy ± $confidence bits")
    
    expected_bias = -(input_dim^seq_len - 1) / (2 * total_samples*log(2))
    println("Expected bias: $expected_bias bits")
    
    # Calculate difference
    diff = abs(empirical_entropy - mc_entropy)
    println("Difference: $diff bits")

    expected_diff = abs(empirical_entropy-expected_bias - mc_entropy)
    println("Expected difference: $expected_diff bits")


    if expected_diff <= confidence
        println("✓ Empirical entropy is within 95% confidence interval of Monte Carlo estimate")
    else
        println("✗ Empirical entropy differs from Monte Carlo estimate")
    end
    
    # Return results for further analysis
    return empirical_entropy, mc_entropy, confidence
end

"""
Extra utility: Plot entropy vs sequence length
"""
function plot_entropy_scaling(model, p_a; 
                             input_dim=2, 
                             max_seq_len=15,
                             n_samples=5000)
    # Only run if we have plotting capabilities
    if !@isdefined(plot)
        println("Plotting not available, skipping entropy scaling analysis")
        return nothing
    end
    
    println("\n=== Analyzing Entropy Scaling with Sequence Length ===")
    
    seq_lengths = 1:max_seq_len
    entropies = Float64[]
    confidences = Float64[]
    
    for seq_len in seq_lengths
        println("Calculating entropy for sequence length $seq_len...")
        
        # For small spaces, use exact calculation
        if input_dim^seq_len <= 10000
            entropy = exact_entropy(model, p_a, seq_len, input_dim, a_range, n_a_points; show_progress=false)
            push!(entropies, entropy)
            push!(confidences, 0.0)
        else
            # For larger spaces, use Monte Carlo
            entropy, confidence = monte_carlo_entropy(
                model, p_a, n_samples, seq_len, input_dim, a_range, n_a_points; show_progress=false
            )
            push!(entropies, entropy)
            push!(confidences, confidence)
        end
    end
    
    # Plot entropy vs sequence length
    p = plot(seq_lengths, entropies,
        title="Entropy vs Sequence Length (dim=$input_dim)",
        xlabel="Sequence Length",
        ylabel="Entropy (bits)",
        marker=:circle,
        ribbon=confidences,
        fillalpha=0.3,
        label="Entropy ± 95% CI"
    )
    
    # Add theoretical maximum entropy line
    max_entropies = [seq_len * log2(input_dim) for seq_len in seq_lengths]
    plot!(p, seq_lengths, max_entropies,
        linestyle=:dash,
        color=:red,
        label="Max Entropy (log₂($input_dim)ᴸ)"
    )
    
    display(p)
    
    # Return results
    return seq_lengths, entropies, confidences
end


# Run the test
#test_calculate_p_b()
#print(test_calculate_entropy_b())
#test_sample_joint_distribution()

# Below is code to initialize the model and p(a) for the debug_entropy_discrepancy.
model_path = "data/inputs/production_generation/models/generated_production_generation.bson"
model = BSON.load(model_path)[:model] |> gpu
p_a = MixtureGaussianPDF([0.5, 0.5], [-0.2, 0.2], [0.1, 0.1])

# Run the debugging analysis
# Run the visualization

function check_specific_sequences(model, p_a; n_samples=100000)
    println("=== Checking Specific Sequence Probabilities ===")
    
    a_val = 0.0
    a_batch = reshape(fill(a_val, n_samples), 1, 1, :) |> gpu
    
    # Generate samples
    samples = generate_samples(model, 2, 5, n_samples, a_batch) |> cpu
    
    # Count specific sequences
    sequence_counts = Dict{String, Int}()
    
    for i in 1:n_samples
        # Convert one-hot to sequence string
        seq = [samples[1, j, i] == 1 ? 1 : 2 for j in 1:5]
        seq_str = join(seq)
        sequence_counts[seq_str] = get(sequence_counts, seq_str, 0) + 1
    end
    
    # Compare with true probabilities for some key sequences
    test_sequences = ["21111", "11222", "22222", "12212", "21212"]
    
    println("Sequence | Empirical | True p(b|a) | Ratio")
    println("---------|-----------|-------------|-------")
    
    for seq_str in test_sequences
        seq = [parse(Int, c) for c in seq_str]
        
        # Get true probability
        seq_onehot = Int.(reshape(Flux.onehotbatch(seq, 1:2), (2, 5, 1))) |> gpu
        a_tensor = reshape([a_val], 1, 1, 1) |> gpu
        true_prob = calculate_p_b_given_a(model, seq_onehot, a_tensor)[1]
        
        # Get empirical probability
        count = get(sequence_counts, seq_str, 0)
        emp_prob = count / n_samples
        
        ratio = emp_prob / true_prob
        println("$seq_str  | $(round(emp_prob, digits=5)) | $(round(true_prob, digits=5)) | $(round(ratio, digits=3))")
    end
    
    # Calculate total variation distance for this specific a
    total_var = 0.0
    all_seqs = Set(keys(sequence_counts))
    
    # Add any sequences that should exist but weren't sampled
    for idx in 0:31
        seq = [((idx >> i) & 1) + 1 for i in 0:4]
        all_seqs = union(all_seqs, [join(seq)])
    end
    
    for seq_str in all_seqs
        seq = [parse(Int, c) for c in seq_str]
        seq_onehot = Int.(reshape(Flux.onehotbatch(seq, 1:2), (2, 5, 1))) |> gpu
        a_tensor = reshape([a_val], 1, 1, 1) |> gpu
        true_prob = calculate_p_b_given_a(model, seq_onehot, a_tensor)[1]
        
        emp_prob = get(sequence_counts, seq_str, 0) / n_samples
        total_var += abs(emp_prob - true_prob)
    end
    
    println("\nTotal variation distance for a=$a_val: $(total_var/2)")
end

check_specific_sequences(model, p_a; n_samples=100000)