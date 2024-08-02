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

function gen_samples(
    psi, numSamples, L; 
    thermalizationSweeps = 500, 
    probabilityFunction = wvfct_sq, 
    eps = 0.,
    nCycles = 1, # can be used to do several cycles of updates, in order to reduce autocorrelation times
  )
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
    for numSample in 1:numSamples
      # global spinflip every 200th step, need to do thermalization again afterwards, probably not necessary to use at all
      # if numSample % 200 == 0
      #  for idx in 1:L
      #    spin = spinflip(spin, idx)
      #  end
      #  for _ in 1:thermalizationSweeps
      #    for idx in 1:L
      #      updatedSpin = spinflip(spin, idx)
      #      newProb = probabilityFunction(psi, updatedSpin; eps)
      #  
      #      rand() > min(1, newProb/prob) && continue
      #  
      #      prob = newProb
      #      spin = updatedSpin
      #    end
      #  end
      # end
 
      for _ in 1:nCycles 
        # iterate through the chain
        for idx in 1:L
          updatedSpin = spinflip(spin, idx)
          newProb = probabilityFunction(psi, updatedSpin; eps)

          rand() > min(1, newProb/prob) && continue

          prob = newProb
          spin = updatedSpin
        end
        push!(samples, spin)
      end
    end

    return samples
    
    
end
