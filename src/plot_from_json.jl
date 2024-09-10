using JSON3
using Plots
using DataFrames

# Function to read and parse JSON file
function read_json_file(file_path)
    open(file_path, "r") do io
        return JSON3.read(io, Dict)
    end
end

# Function to extract data from JSON structure
function extract_data(json_data)
    experiments = json_data["experiments"]
    data = DataFrame()
    for (key, value) in experiments
        row = Dict{String, Any}()
        merge!(row, value["parameters"])
        merge!(row, value["experiment_modes"])
        merge!(row, value["hyperparameters"])
        merge!(row, value["results"])
        merge!(row, value["other_results"])
        push!(data, row, cols=:union)
    end
    return data
end

# Function to get unique configurations
function get_unique_configs(data, exclude_cols)
    config_cols = setdiff(names(data), exclude_cols)
    unique_configs = unique(data[:, config_cols])
    return unique_configs
end

# Function to create and save plot
function create_and_save_plot(data, x_col, y_cols, config, other_vars, saving_path, log_scale)
    plot_data = data
    for (col, val) in pairs(config)
        plot_data = plot_data[plot_data[!, col] .== val, :]
    end
    plot_data = sort(plot_data, x_col)
    
    config_str = join(["$(k)=$(v)" for (k,v) in pairs(config)], "_")

    p = plot(xlabel=x_col, ylabel="Values",title = config_str, titlefont = font(11), guidefont = font(10))

    if log_scale
        plot!(xaxis=:log2)
    end
    
    # Group by other_vars
    if !isempty(other_vars)
        grouped = collect(pairs(groupby(plot_data, other_vars)))
    else
        grouped = [(NamedTuple(), plot_data)]
    end
    
    # Color and marker combinations
    colors = [:blue, :red, :green, :orange, :purple, :cyan, :magenta, :yellow]
    markers = [:circle, :square, :diamond, :triangle, :cross, :hexagon, :star5, :star8]
    
    for (i, (group_key, group_data)) in enumerate(grouped)
        color = colors[mod1(i, length(colors))]
        marker = markers[mod1(i, length(markers))]
        
        for (j, y_col) in enumerate(y_cols)
            label = if !isempty(other_vars)
                group_label = join(["$k=$v" for (k,v) in pairs(group_key)], ", ")
                "$y_col ($group_label)"
            else
                y_col
            end
            
            plot!(p, group_data[!, x_col], group_data[!, y_col], 
                  label=label, color=color, marker=marker, 
                  linestyle=j == 1 ? :solid : :dash)
        end
    end
   
    plot!(p, margin=5Plots.mm, legend=:outerbottom)  
    foldername = "$(x_col)_vs_$(join(y_cols, "_"))"
    filename = "$(config_str).pdf"
    try 
        mkpath(joinpath(saving_path, foldername)) 
    catch e 
        @warn "Probably folder already exists: " * string(e)
    end
    savefig(p, joinpath(saving_path, foldername, filename))
    println("Saved plot: $filename")
end

# Main function
function custom_plot(configdict)


    file_path = joinpath("data", "outputs", configdict["path"])
    json_file_path = joinpath(file_path, "results", "result.json")
    
    # Read JSON file
    json_data = read_json_file(json_file_path)
    
    # Extract data
    data = extract_data(json_data)

    # The result columns:
    r_cols = ["entropy", "conditional_entropy", "mutual_information", "avg_time_conditional_entropy", "avg_time_entropy"]
    
    x_col = configdict["x"]
    
    log_scale = configdict["log_scale"]
    log_scale = (log_scale == "y")


    y_cols = configdict["y"]

    other_vars = configdict["other_vars"]

    saving_path = joinpath(file_path, "special_plots")
    
    try mkpath(saving_path) catch e @warn "Probably file already exists: " * e.msg end
    
    # Get unique configurations
    exclude_cols = vcat([x_col], r_cols, other_vars)
    unique_configs = get_unique_configs(data, exclude_cols)
    
    # Create plots for each unique configuration
    for config in eachrow(unique_configs)
        create_and_save_plot(data, x_col, y_cols, config, other_vars, saving_path, log_scale)
    end
    
    println("All plots have been saved.")
end

# Run the main function
# configdict = Dict("path" => "_old/transfer_time_evolve_convergence_large_batch256_v1", "x" => "t", "y" => ["entropy", "conditional_entropy"], "other_vars" => ["num_samples","g"], "log_scale" => "y")
# 
# custom_plot(configdict)

configdict = Dict("path" => "merged_discrete_noise_results_all_with_no_noise", "x" => "num_samples", "y" => ["entropy","conditional_entropy"], "other_vars" => [], "log_scale" => "y")
custom_plot(configdict)

config2 = Dict("path" => "merged_discrete_noise_results_all_with_no_noise", "x" => "num_samples", "y" => ["mutual_information"], "other_vars" => [], "log_scale" => "y")
custom_plot(config2)
