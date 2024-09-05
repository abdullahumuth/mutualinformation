using CSV
using DataFrames
using JSON3
using Dates
using OrderedCollections

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
    output_file = joinpath(folder_path, "unified_experiments.json")
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
        mv(joinpath(folder_path, file), joinpath(folder_path, "old", file))
    end
end

foreach((d = readdir("data/old", join=true); d[isdir.(d)])) do folder
    if !isdir(joinpath(folder, "results/old"))
        nothing
    else
        csv_to_json_converter("$folder/results")
        #move_csv("$folder/results/")
    end
end



# Usage example:
# csv_to_json_converter("path/to/your/csv/folder")
#csv_to_json_converter("data/old/sample_convergence_newest_batch512_v1/results/")
#move_csv("data/old/sample_convergence_newest_batch512_v1/results/")