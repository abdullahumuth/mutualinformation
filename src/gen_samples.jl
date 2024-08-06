### prob distributions ###
function sample_to_int(conf)
  conf = 0.5 .* (conf .+ 1)
  pos = mapreduce(+, enumerate(conf)) do (l,s)
      return s*2^(l-1)
  end
  return floor(Int, pos + 1)
end

function wvfct_sq(psi, sample; eps=1e-3)
  return abs2(psi[sample_to_int(sample)]) ./ sum(abs2, psi)
end

function importance_prob(psi, sample; eps=1e-3)
  p = abs2(psi[sample_to_int(sample)])
  maxPsi = findmax(abs2.(psi))[1]
  
  return p/maxPsi < eps ? eps : p
end

function uniform(psi, sample; eps=1e-3)
  return 1/(2^length(sample))
end

function spinflip(spin::Vector{Int64}, pos::Int)
  res = copy(spin)
  res[pos] *= -1 
  return res
end

function gen_samples(psi, num_samples, L; thermalizationSweeps = 500, probabilityFunction = wvfct_sq, eps = 0)
    spin = fill(1, L)
    prob = probabilityFunction(psi, spin; eps)

    for _ in 1:thermalizationSweeps
      for idx in 1:L
        updatedSpin = spinflip(spin, idx)
        newProb = probabilityFunction(psi, updatedSpin; eps)

        rand() > min(1, newProb/prob) && continue

        prob = newProb
        spin = updatedSpin
      end
    end

    samples = []
    for numSample in 1:num_samples
 
      # iterate through the chain
      for idx in 1:L
        updatedSpin = spinflip(spin, idx)
        newProb = probabilityFunction(psi, updatedSpin; eps)

        rand() > min(1, newProb/prob) && continue

        prob = newProb
        spin = updatedSpin
      end

      idx = rand(1:L)
      updatedSpin = spinflip(spin, idx)
      newProb = probabilityFunction(psi, updatedSpin; eps)
      if rand() < min(1, newProb/prob)
          prob = newProb
          spin = updatedSpin
      end
      push!(samples, spin)
    end

    return samples
end