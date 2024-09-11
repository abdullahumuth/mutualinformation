using CSV
using DataFrames
using JSON3
using Dates
using OrderedCollections
include("main.jl")


function csv_to_json_converter(folder_path)
    # Get all CSV files in the folder
    csv_files = filter(f -> endswith(f, ".csv"), readdir(folder_path * "/old", join=true))
    
    # Initialize an array to hold all experiment data
    all_experiments = OrderedDict()
    
    for file in csv_files
        # Read the CSV file
        df = CSV.read(file, DataFrame)
        
        # Extract the experiment name and parameters from the file name
        file_name = basename(file)
        parts = split(replace(file_name, r".csv$" => ""), "_")
        
        # Initialize dictionaries for parameters and hyperparameters
        parameters = OrderedDict()
        hyperparameters = OrderedDict()
        experiment_modes = OrderedDict()
        
        # Parse the file name
        for part in parts
            a = split(part, "=")
            if length(a) < 2
                continue
            end
            key, value = a[1], a[2]
            if key == "samples"
                key = "num_samples"
            end
            if key in ["L", "J", "g", "t", "num_samples"]
                parameters[key] = parse(Float64, value)
            else
                # Assume any other parameter is a hyperparameter
                hyperparameters[key] = try
                    parse(Float64, value)
                catch
                    value  # Keep as string if not a number
                end
            end
        end

        # Create the experiment structure
        all_experiments[file_name] = OrderedDict(
            "parameters" => parameters,
            "experiment_modes" => experiment_modes,
            "hyperparameters" => hyperparameters,
            "results" => Dict(),
            "other_results" => Dict()
        )
        
        # Populate the experiment structure from the DataFrame
        for col in names(df)
            value = df[1, col]  # Assuming each CSV has only one row of data
            if col in ["mutual_information", "entropy", "conditional_entropy"]
                all_experiments[file_name]["results"][col] = value
            elseif startswith(col, "avg_time_")
                all_experiments[file_name]["other_results"][col] = value
            else
                # Any column not explicitly handled goes into hyperparameters
                all_experiments[file_name]["hyperparameters"][col] = value
            end
        end

    end
    
     # Define a custom sorting function
     function experiment_sorter(exp)
        #println(exp)
        # First, sort by L, J, g, t
        primary_sort = sort([
            (k, v) for (k, v) in exp["parameters"]
        ])
        
        # Finally, sort by other hyperparameters
        secondary_sort = sort([
            (k, v) for (k, v) in exp["hyperparameters"]
        ])
        
        return (primary_sort, secondary_sort)
    end
    
    # Sort the experiments using the custom sorting function
    sort!(all_experiments, by = experiment_sorter ;byvalue = true)

    # Create the final JSON structure
    json_data = OrderedDict(
        "experiments" => all_experiments,
        "metadata" => OrderedDict(
            "conversion_date" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
            "number_of_experiments" => length(all_experiments)
        )
    )
    
    # Write to a JSON file
    output_file = joinpath(folder_path, "result.json")
    open(output_file, "w") do io
        JSON3.pretty(io, json_data)
    end
    
    println("Conversion complete. Output file: $output_file")
end

function move_csv(folder_path)
    csv_files = filter(f -> endswith(f, ".csv"), readdir(folder_path))
    if !isdir(joinpath(folder_path, "old"))
        mkdir(joinpath(folder_path, "old"))

    end
    for file in csv_files
        mv(joinpath(folder_path, file), joinpath(folder_path, "old", file), force=true)
    end
end


foreach((d = readdir("data/outputs", join=true); d[isdir.(d)])) do folder
    if !(folder == "data/outputs\\_old" || folder == "data/outputs\\_logs")
        move_csv("$folder/results/")
        if !isfile(joinpath(folder, "results/result.json"))
            csv_to_json_converter("$folder/results")
        end
        if isfile(joinpath(folder, "results/unified_experiments.json"))
            rm(joinpath(folder, "results/unified_experiments.json"))
        end
        if isfile(joinpath(folder, "results/old/unified_experiments.json"))
            rm(joinpath(folder, "results/old/unified_experiments.json"))
        end
    end
end




function merge_json_files(file1, file2, output_file)
    # Read the JSON data from the two files
    data1 = copy(JSON3.read(read(file1, String)))[:experiments]
    data2 = copy(JSON3.read(read(file2, String)))[:experiments]

    # Merge the two datasets
    merged_data = merge(data1, data2)

    output_json(merged_data, output_file)
end

# Example usage
# merge_json_files("data/outputs/noise0.1_discrete_sample_v1/results/result.json", "data/outputs/noise0.01_discrete_sample_v1/results/result.json", "merged_results")
merge_json_files("data/outputs/merged_discrete_noise_results_all/results/result.json", "data/outputs/_old/ER_improved_sample_convergence_l20_batch128_v1/results/result.json", "merged_discrete_noise_results_all_with_no_noise_v2")
merge_json_files("data/outputs/merged_discrete_noise_results_all_with_no_noise_v2/results/result.json", "data/outputs/noiseless_L12_discrete_sample_v1/results/result.json", "merged_discrete_noise_results_all_with_no_noise_l12")

# Usage example:
# csv_to_json_converter("path/to/your/csv/folder")
# csv_to_json_converter("data/old/sample_convergence_newest_batch512_v1/results/")
# move_csv("data/old/sample_convergence_newest_batch512_v1/results/")


function deep_convert_to_dict(obj)
    if obj isa AbstractDict
        return OrderedDict(k => deep_convert_to_dict(v) for (k, v) in obj)
    elseif obj isa AbstractArray
        return [deep_convert_to_dict(v) for v in obj]
    else
        return obj
    end
end

function modify_json(input_file, output_file)
    # Read the JSON data
    json_data = JSON3.read(read(input_file, String))

    # Convert to a fully mutable structure
    mutable_data = deep_convert_to_dict(json_data)

    # Iterate through each experiment in the JSON
    for (key, value) in mutable_data[:experiments]
        # Check if the filename doesn't contain "noise="
        if occursin(r"noise=0.001", string(key))
            # If "experiment_modes" doesn't exist, create it
            if !haskey(value, :experiment_modes)
                value[:experiment_modes] = OrderedDict()
            end
            
            # Add "noise": 0 to experiment_modes
            value[:experiment_modes][:noise] = 0.001
        end
    end

    # Write the modified data back to a new JSON file
    open(output_file, "w") do io
        JSON3.pretty(io, mutable_data)
    end
end
# Example usage
# input_file = "data/outputs/merged_discrete_noise_results_all_with_no_noise/results/resulting3.json"
# output_file = "data/outputs/merged_discrete_noise_results_all_with_no_noise/results/resulting4.json"
# modify_json(input_file, output_file)


