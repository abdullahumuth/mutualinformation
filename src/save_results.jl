using Plots
using BSON
using CSV
using DataFrames
using Distributions
using HDF5
using OrderedCollections
using JSON3
using Dates

function mkdir_safe(path)
    try mkdir(path) catch e @warn "Probably file already exists: " * e.msg end
end



function save_results(entropy, conditional_entropy, name, data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)

    if !isdir("data/outputs") mkdir("data/outputs") end
    if !isdir("data/outputs/$(name)") mkdir_safe("data/outputs/$(name)") end
    if !isdir("data/outputs/$(name)/models") mkdir_safe("data/outputs/$(name)/models") end
    if !isdir("data/outputs/$(name)/losses") mkdir_safe("data/outputs/$(name)/losses") end
    if !isdir("data/outputs/$(name)/plots") mkdir_safe("data/outputs/$(name)/plots") end
    if !isdir("data/outputs/$(name)/results") mkdir_safe("data/outputs/$(name)/results") end



    if !isfile("data/outputs/$name/results/result.json")
        try
            output_json(Dict(), name)
        catch e
            if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
            @warn "Failed to create JSON " * str
        end
    end
        
    


    file_name = name_files(merge(data_gen_param_dict, exp_modes_param_dict, hyper_param_dict))

    try
        save_models(entropy, conditional_entropy, name, file_name)
    catch e
        if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
        @warn "Failed to save models: " * str
    end


    try
       output_loss(entropy, conditional_entropy, name, file_name)
    catch e
        if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
        @warn "Failed to save csv: " * str
    end


    try
        json = read("data/outputs/$name/results/result.json", String)
        experiments = copy(JSON3.read(json))[:experiments]
        experiments[Symbol(file_name)] = output(entropy, conditional_entropy, data_gen_param_dict, exp_modes_param_dict, hyper_param_dict)
        output_json(experiments, name)
    catch e
        if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
        @warn "Failed to save JSON " * str
    end
    

    try
        save_plots(entropy, conditional_entropy, name, file_name)
    catch e
        if "msg" in fieldnames(typeof(e)) str = e.msg else str = "No message" end
        @warn "Failed to save plots: " * str
    end
end


function plot_models(a, b)
    p = plot(a[2].train_losses, label="train", title="Entropy= $(a[1])  (min. $(a[2].min_epoch))")
    plot!(a[2].test_losses, label="test")
    p2 = plot(b[2].train_losses, label="train", title="Conditional Entropy= $(b[1])  (min. $(b[2].min_epoch))")
    plot!(b[2].test_losses, label="test")
    return plot(p,p2, layout = (2,1))
end


function name_files(dict)
    return join(["$(k)=$(v)_" for (k,v) in zip(keys(dict), values(dict))])
end

function save_models(a, b, exp_name, file_name)
    let 
        entropy_model = cpu(a[2].net)
        conditional_entropy_model = cpu(b[2].net)
        bson("data/outputs/$exp_name/models/" * file_name * ".bson", entropy = entropy_model, conditional_entropy = conditional_entropy_model)
    end
end

function output(a, b, params, exp_params, hyper_params)

    # Construct the data structure
    data = Dict(
        "parameters" => params,
        "experiment_modes" => exp_params, 
        "hyperparameters" => hyper_params,
        "results" => Dict(
            "mutual_information" => a[1] - b[1],
            "entropy" => a[1],
            "conditional_entropy" => b[1]
        ),
        "other_results" => Dict(
            "avg_time_entropy" => a[2].avg_time,
            "avg_time_conditional_entropy" => b[2].avg_time
        )
    )
    return data
end


function output_json(data, exp_name)
    # Create the directory if it doesn't exist
    try mkpath("data/outputs/$exp_name/results") catch e end

    # Generate the filename
    filename = "data/outputs/$exp_name/results/result.json"

    


    # Write to JSON file
    open(filename, "w") do io
        JSON3.pretty(io, OrderedDict("experiments" => data,
        "metadata" => OrderedDict(
            "date" => Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"),
            "number_of_experiments" => length(data)
        )))
    end
end
    


function output_loss(a,b, exp_name, file_name)
    entropy_loss = DataFrame(epoch = 1:length(a[2].train_losses), train = a[2].train_losses, test = a[2].test_losses)
    CSV.write("data/outputs/$exp_name/losses/entropy_" * file_name * ".csv", entropy_loss)
    conditional_entropy_loss = DataFrame(epoch = 1:length(b[2].train_losses), train = b[2].train_losses, test = b[2].test_losses)
    CSV.write("data/outputs/$exp_name/losses/conditional_entropy_" * file_name * ".csv", conditional_entropy_loss)
end

function save_plots(a,b,exp_name, file_name)
    savefig(plot_models(a,b), "data/outputs/$exp_name/plots/" * file_name * ".png")
end
