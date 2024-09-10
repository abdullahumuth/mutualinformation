function mkdir_safe(path)
    try mkdir(path) catch e @warn "Probably file already exists: " * e.msg end
end


