include("./read_wvfct.jl")
using Plots
using BSON
using CSV
using DataFrames
using Distributions
using HDF5
using OrderedCollections
using JSON3
using Dates


function create_dataset(L,J,g,t,num_samples; noise=0, load = "", discrete=true, uniform = false, unique = false, fake = false, shuffle=false)
    if load != ""
        data = load_generated_data(load, num_samples, discrete)
        return data.data |> gpu
    end
    if L == 20
        psi = read_wavefunction(L, J, g, t)
    else
        psi = read_hdf5_wavefunction(L, J, g, t)
    end
    if unique
        indices = randperm(Xoshiro(303), 2^L)[1:num_samples]
    else
        if uniform
            dist = DiscreteUniform(1, 2^L)
        else
            dist = Categorical(abs2.(psi))
        end
        indices = rand(Xoshiro(303), dist, num_samples)
    end

    f(x) = digits(x, base=2, pad = L)|> reverse
    x_proto = stack(map(f, indices .- 1))

    x = zeros((2, size(x_proto)...))
    x[1, :, :] .= x_proto
    x[2, :, :] .= 1 .- x_proto
    x = Int.(x) |> gpu


    if fake
        y = fake_y(L; unit=1, offset=16, partition=2, shuffle=shuffle)[:, indices]
    else
        y = stack(map(x -> [real(psi[x]), imag(psi[x])], indices))
    end
    noise > 0 && (y .+= randn(Xoshiro(303), size(y)) .* noise)
    y = reshape(y, (1, size(y)...)) |> gpu
    discrete && (return x, y)
    return y, x
end


function fake_y(n; unit=1, offset=10, partition=1, magnetization=true, shuffle=true)
    f(x) = [div((x-1),sqrt(2^n))*unit,((x-1) % sqrt(2^n))*unit] .+ unit*(-2^(n/2-1) .+ 1/2)
    
    magnetization && (shuffled = invperm(sort(1:2^n, lt=compare_integers)))
    shuffle && (Random.randperm!(MersenneTwister(111), shuffled))
    if partition == 1
      straight = stack(map(f, 1:2^n))
      g = straight[:,shuffled]
    elseif partition == 2
      h1 = stack(map(f, 1:2^(n-1))) .+ [offset, 0]
      h2 = stack(map(f, 1:2^(n-1))) .+ [-offset, 0]
      g = hcat(h1, h2)[:, shuffled]
    elseif partition == 4
      h1 = stack(map(f, 1:2^(n-2))) .+ [offset, offset]
      h2 = stack(map(f, 1:2^(n-2))) .+ [-offset, offset]
      h3 = stack(map(f, 1:2^(n-2))) .+ [offset, -offset]
      h4 = stack(map(f, 1:2^(n-2))) .+ [-offset, -offset]
      g = hcat(h1, h2, h3, h4)[:, shuffled]
    end
  
    g = g./sqrt(sum((x)->x^2,g))
    return g
  end
  
  function compare_integers(a::Integer, b::Integer)
    sum_a = count_ones(a-1)
    sum_b = count_ones(b-1)
    if sum_a == sum_b
        return false
    elseif sum_a > sum_b
        return true
    else
        return false
    end
  end
  